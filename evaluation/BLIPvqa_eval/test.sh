export CUDA_VISIBLE_DEVICES=6

export project_dir="BLIPvqa_eval/"

## ------------------------------------------------------------------
##              Evaluation on AE set    
## ------------------------------------------------------------------


cd $project_dir
categories="animals animals_objects objects"
# out_dir="../examples/"
for category in $categories
do
    out_dir="../outputs/output_eval_full/aeprompts/SDXLCorrector/cfgpp0.8_LatentCorr_step5_modCompWts_beta0.9_algouflow_same_t_sum1_uncondcorr_hybrid_6_nresmpl_4_/${category}/"
    python BLIP_vqa.py --out_dir=$out_dir --np_num=4
done


## ------------------------------------------------------------------
##              Evaluation multi_concept set   
## ------------------------------------------------------------------


# cd $project_dir
# categories="3_concepts 4_concepts 5_concepts 6_concepts 7_concepts"
# # out_dir="../examples/"
# for category in $categories
# do
#     out_dir="../outputs/output_eval_full/aeprompts/SDXLCorrector/cfgpp0.8_LatentCorr_step5_modCompWts_beta0.9_algouflow_same_t_sum1_uncondcorr_hybrid_6_nresmpl_3_/${category}/"
#     python BLIP_vqa.py --out_dir=$out_dir --np_num=4
# done




## ------------------------------------------------------------------
##              Evaluation on t2icompbench set    
## ------------------------------------------------------------------


# cd $project_dir
# categories="color_val shape_val texture_val" 
# # out_dir="../examples/"
# for category in $categories
# do
#     out_dir="../outputs/output_eval_full/t2icompbench/SDXLCorrector/cfgpp0.8_LatentCorr_step5_algosame_t_4_/${category}/"
#     python BLIP_vqa.py --out_dir=$out_dir --np_num=8
# done

