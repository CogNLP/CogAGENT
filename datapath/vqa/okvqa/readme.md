# okvqa

## download from

https://github.com/microsoft/PICa

## include

coco_annotations
--captions_train2014.json
--captions_val2014.json
--mscoco_train2014_annotations.json
--mscoco_val2014_annotations.json
--OpenEnded_mscoco_train2014_questions.json
--OpenEnded_mscoco_val2014_questions.json

coco_clip_new
--okvqa_qa_line2sample_idx_train2014.json
--okvqa_qa_line2sample_idx_val2014.json

input_text
--coco_caption_pred_tags
----test.score.json.lineidx
----test.score.json.tsv
----train.score.json.lineidx
----train.score.json.tsv
----val.score.json.lineidx
----val.score.json.tsv
--vinvl_caption
----VinVL_base_test.tsv
----VinVL_base_val.tsv
----VinVL_base_val2014.tsv

## note

We need to run toolkit.prepare() to replace following data

in under file okvqa/coco_clip_newnew
coco_clip_vitb16_train2014_okvqa_convertedidx_image.npy
coco_clip_vitb16_train2014_okvqa_question.npy
