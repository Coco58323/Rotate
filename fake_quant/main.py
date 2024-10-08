import utils
import torch
import model_utils
import data_utils
import transformers
import quant_utils
import rotation_utils
import gptq_utils
import eval_utils
import hadamard_utils
import smooth_utils

def main():
    args = utils.parser_gen()
    if args.wandb:
        import wandb
        wandb.init(project=args.wandb_project, name=args.wandb_id)
        wandb.config.update(args)
        
    transformers.set_seed(args.seed)
    model = model_utils.get_model(args.model, args.hf_token)
    model.eval()
    if model.lm_head.weight.device == torch.device('meta'):
        import torch.nn as nn
        model.lm_head.weight = nn.Parameter(model.model.embed_tokens.weight.clone())
    
    if args.smooth:
        scales = torch.load(args.smooth)
        smooth_utils.smooth_lm(model, scales, 0.5)

    # Rotate the weights
    if args.rotate:
        rotation_utils.fuse_layer_norms(model)
        rotation_utils.rotate_model(model, args)
        utils.cleanup_memory(verbos=True)
            
        quant_utils.add_actquant(model) #Add Activation Wrapper to the model
        qlayers = quant_utils.find_qlayers(model)
        for name in qlayers:
            if 'o_proj' in name and not args.offline:
                had_K, K = hadamard_utils.get_hadK(model.config.num_attention_heads)
                qlayers[name].online_partial_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].had_dim = model.config.hidden_size//model.config.num_attention_heads
                qlayers[name].fp32_had = args.fp32_had
            if ('down_proj' in name or 'w2' in name) and (f'layers.{args.target}.' in name or args.target == -1) and not args.offline:
                had_K, K = hadamard_utils.get_hadK(model.config.intermediate_size)
                qlayers[name].online_full_had = True
                qlayers[name].had_K = had_K
                qlayers[name].K = K
                qlayers[name].fp32_had = args.fp32_had
    else:
        quant_utils.add_actquant(model) #Add Activation Wrapper to the model as the rest of the code assumes it is present
        
                
    if args.w_bits < 16:
        save_dict = {}
        if args.load_qmodel_path: # Load Quantized Rotated Model
            assert args.rotate, "Model should be rotated to load a quantized model!"
            assert not args.save_qmodel_path, "Cannot save a quantized model if it is already loaded!"
            print("Load quantized model from ", args.load_qmodel_path)
            save_dict = torch.load(args.load_qmodel_path)
            model.load_state_dict(save_dict["model"])
            
        elif not args.w_rtn: # GPTQ Weight Quantization
            # assert "llama" in args.model, "Only llama is supported for GPTQ!"
            
            trainloader = data_utils.get_loaders(
                args.cal_dataset, nsamples=args.nsamples,
                seed=args.seed, model=args.model,
                seqlen=model.seqlen, eval_mode=False
            )
            quantizers = gptq_utils.gptq_fwrd(model, trainloader, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers
        else: # RTN Weight Quantization
            quantizers = gptq_utils.rtn_fwrd(model, utils.DEV, args)
            save_dict["w_quantizers"] = quantizers
            
        if args.save_qmodel_path:
            save_dict["model"] = model.state_dict()
            torch.save(save_dict, args.save_qmodel_path)


    # Add Input Quantization
    if args.a_bits < 16 or args.v_bits < 16:
        qlayers = quant_utils.find_qlayers(model, layers=[quant_utils.ActQuantWrapper])
        down_proj_groupsize = -1
        if args.a_groupsize > 0:
            down_proj_groupsize = utils.llama_down_proj_groupsize(model, args.a_groupsize)
        
        for name in qlayers:            
            layer_input_bits = args.a_bits
            layer_groupsize = args.a_groupsize
            layer_a_sym = not(args.a_asym)
            layer_a_clip = args.a_clip_ratio
            
            if 'v_proj' in name and args.v_bits < 16: #Set the v_proj precision
                qlayers[name].out_quantizer.configure(bits=args.v_bits,
                                              groupsize=args.v_groupsize,
                                              sym=not(args.v_asym),
                                              clip_ratio=args.v_clip_ratio)
                qlayers[name].out_runtime_smooth = args.a_runtime_smooth
            
            # if 'k_proj' in name and args.k_bits < 16: #Set the k_proj precision
            #     qlayers[name].out_quantizer.configure(bits=args.k_bits,
            #                                   groupsize=args.k_groupsize,
            #                                   sym=not(args.k_asym),
            #                                   clip_ratio=args.k_clip_ratio)
            #     qlayers[name].out_runtime_smooth = args.a_runtime_smooth
            
            if 'lm_head' in name: #Skip lm_head quantization   
                layer_input_bits = 16
            
            if 'down_proj' in name: #Set the down_proj precision
                if args.int8_down_proj:
                    layer_input_bits = 8
                layer_groupsize = down_proj_groupsize

            # if 'down_proj' not in name:
            if args.exclude_qkv:
                if 'q_proj' not in name and 'k_proj' not in name and 'v_proj' not in name:
                    qlayers[name].runtime_smooth = args.a_runtime_smooth
            elif args.include_o:
                if 'q_proj' not in name and 'k_proj' not in name and 'v_proj' not in name and 'gate_proj' not in name and 'up_proj' not in name and 'down_proj' not in name:
                    qlayers[name].runtime_smooth = args.a_runtime_smooth
            else:
                qlayers[name].runtime_smooth = args.a_runtime_smooth
            qlayers[name].per_tensor = args.a_per_tensor
            qlayers[name].reorder = args.a_reorder
            qlayers[name].act_scale_g128 = args.act_scale_g128
            qlayers[name].scale_groupsize = args.scale_groupsize
            qlayers[name].quant_scales = args.a_quant_scales
            qlayers[name].quantizer.configure(bits=layer_input_bits,
                                              groupsize=layer_groupsize,
                                              sym=layer_a_sym,
                                              clip_ratio=layer_a_clip)

    if args.k_bits < 16:
        if args.k_pre_rope:
            raise NotImplementedError("Pre-RoPE quantization is not supported yet!")
        else:
            rope_function_name = model_utils.get_rope_function_name(model)
            layers = model_utils.get_layers(model)
            k_quant_config = {'k_bits':args.k_bits, "k_groupsize": args.k_groupsize,
                                          "k_sym": not(args.k_asym), "k_clip_ratio": args.k_clip_ratio}
            for layer in layers:
                rotation_utils.add_qk_rotation_wrapper_after_function_call_in_forward(
                            layer.self_attn, 
                            rope_function_name, 
                            config=model.config,
                            **k_quant_config)
        
    # Evaluating on dataset
    testloader = data_utils.get_loaders(
            args.eval_dataset,
            seed=args.seed,
            model=args.model,
            seqlen=model.seqlen,
            hf_token=args.hf_token,
            eval_mode=True
        )
    

    dataset_ppl = eval_utils.evaluator(model, testloader, utils.DEV, args)
    qlayers = quant_utils.find_qlayers(model)
    total = 0
    interval = 3
    prob = {}
    for k in ['v_proj', 'o_proj', 'down_proj', 'up_proj']:
        prob[k] = {}
        for j in ['x', 'r', 'rs', 'rrs']:
            prob[k][j] = [0]*interval
    for name in qlayers:
        if 'q_proj' in name or 'k_proj' in name or 'gate_proj' in name:
            continue
        if 'lm_head' in name:
            continue
        # print(name, qlayers[name].total, qlayers[name].prob_r, qlayers[name].prob_rrs, qlayers[name].prob_x)
        # total += qlayers[name].total
        if 'down_proj' in name:
            key = 'down_proj'
        elif 'up_proj' in name:
            key = 'up_proj'
        elif 'v_proj' in name:
            key = 'v_proj'
        elif 'o_proj' in name:
            key = 'o_proj'
        for j in range(interval):
            prob[key]['x'][j] += qlayers[name].prob_x[j]
            prob[key]['r'][j] += qlayers[name].prob_r[j]
            prob[key]['rrs'][j] += qlayers[name].prob_rrs[j]
            prob[key]['rs'][j] += qlayers[name].prob_rs[j]
    # print("Prob x: ", prob_x)
    # print("Prob R: ", prob_r)
    # print("Prob RS: ", prob_rs)
    # print("Prob RRS: ", prob_rrs)
    # print("Prob Mixed RRS: ", prob_mixed_rrs)
    print(prob)
    if args.wandb:
            wandb.log({'ppl/{}'.format(args.eval_dataset.upper()): dataset_ppl})

    if not args.lm_eval:
        return
    else:
        # Import lm_eval utils
        import lm_eval
        from lm_eval import utils as lm_eval_utils
        from lm_eval.api.registry import ALL_TASKS
        from lm_eval.models.huggingface import HFLM

        
    
    if args.distribute:
        utils.distribute_model(model)
    else:
        model.to(utils.DEV)
    
    tokenizer = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False, use_auth_token=args.hf_token)
    hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size='auto', apply_chat_template=args.apply_chat_template)

    # task_names = lm_eval_utils.pattern_match(args.tasks, ALL_TASKS)
    tasks = []
    from typing import List
    if isinstance(args.tasks, List):
        for task in args.tasks:
            tasks.extend(task.split(','))
    results = lm_eval.simple_evaluate(hflm, tasks=tasks, batch_size='auto')['results']
    metric_vals = {task: round(result.get('acc_norm,none', result['acc,none']), 4) for task, result in results.items()}

    metric_vals['acc_avg'] = round(sum(metric_vals.values()) / len(metric_vals.values()), 4)
    print(metric_vals)
    if args.wandb:
        wandb.log(metric_vals)

if __name__ == '__main__':
    main()
