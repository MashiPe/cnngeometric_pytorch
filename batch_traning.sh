
# # Training models without occlusion

# echo "Training TPS 8P no occlusion"
# python3 train_strong.py --geometric-model tps --num-epochs 20 --training-dataset streetview  --dataset-csv-path training_data/streetview-random --dataset-image-path datasets --feature-extraction-cnn resnet101 --trained-model-dir trained_models --reg-grid False --trained-model-fn tps_8p_no_occ --x-axis-coords -1.0 -1.0 -1.0 0.0 0.0  1.0 1.0 1.0 --y-axis-coords -1.0 0.0 1.0 -1.0 1.0 -1.0 0.0 1.0
# echo "Training TPS 4P no occlusion"
# python3 train_strong.py --geometric-model tps --num-epochs 20 --training-dataset streetview  --dataset-csv-path training_data/streetview-random --dataset-image-path datasets --feature-extraction-cnn resnet101 --trained-model-dir trained_models --reg-grid False --trained-model-fn tps_4p_no_occ --x-axis-coords -1.0 -1.0 1.0 1.0 --y-axis-coords -1.0 1.0 -1.0 1.0
# echo "Training TPS 9P no occlusion"
# python3 train_strong.py --geometric-model tps --num-epochs 20 --training-dataset streetview --dataset-csv-path training_data/streetview-random --dataset-image-path datasets --feature-extraction-cnn resnet101 --trained-model-dir trained_models --reg-grid True --trained-model-fn tps_9p_no_occ
# echo "Training Affine no occlusion"
# python3 train_strong.py --geometric-model affine --num-epochs 20 --training-dataset streetview  --dataset-csv-path training_data/streetview-random --dataset-image-path datasets --feature-extraction-cnn resnet101 --trained-model-dir trained_models --trained-model-fn affine_no_occ
echo "Training Affine no occlusion only translation"
python3 train_strong.py --geometric-model affine --translation-only True --num-epochs 20 --training-dataset streetview  --dataset-csv-path training_data/streetview-random --dataset-image-path datasets --feature-extraction-cnn resnet101 --trained-model-dir trained_models --trained-model-fn affine_no_occ_translation
# echo "Training Hom no occlusion"
# python3 train_strong.py --geometric-model hom --num-epochs 20 --training-dataset streetview --dataset-csv-path training_data/streetview-random --dataset-image-path datasets --feature-extraction-cnn resnet101 --trained-model-dir trained_models --trained-model-fn hom_no_occ


# # Training models with occlusion

# echo "Training TPS 8P occlusion"
# python3 train_strong.py --geometric-model tps --num-epochs 21 --training-dataset streetview --log_interval 50 --dataset-csv-path training_data/streetview-random --dataset-image-path datasets --feature-extraction-cnn resnet101 --trained-model-dir trained_models --reg-grid False --partial-occlusion True --trained-model-fn tps_8p_occ --x-axis-coords -1.0 -1.0 -1.0 0.0 0.0  1.0 1.0 1.0 --y-axis-coords -1.0 0.0 1.0 -1.0 1.0 -1.0 0.0 1.0
# echo "Training TPS 4P occlusion"
# python3 train_strong.py --geometric-model tps --num-epochs 20 --training-dataset streetview  --dataset-csv-path training_data/streetview-random --dataset-image-path datasets --feature-extraction-cnn resnet101 --trained-model-dir trained_models --reg-grid False --partial-occlusion True --trained-model-fn tps_4p_occ --x-axis-coords -1.0 -1.0 1.0 1.0 --y-axis-coords -1.0 1.0 -1.0 1.0
# echo "Training TPS 9P occlusion"
# python3 train_strong.py --geometric-model tps --num-epochs 20 --training-dataset streetview --log_interval 50 --dataset-csv-path training_data/streetview-random --dataset-image-path datasets --feature-extraction-cnn resnet101 --trained-model-dir trained_models --reg-grid True  --partial-occlusion True --trained-model-fn tps_9p_occ
# echo "Training Affine occlusion"
# python3 train_strong.py --geometric-model affine --num-epochs 20 --training-dataset streetview --log_interval 50 --dataset-csv-path training_data/streetview-random --dataset-image-path datasets --feature-extraction-cnn resnet101 --trained-model-dir trained_models  --partial-occlusion True --trained-model-fn affine_occ
# echo "Training Hom occlusion"
# python3 train_strong.py --geometric-model hom --num-epochs 20 --training-dataset streetview --log_interval 50 --dataset-csv-path training_data/streetview-random --dataset-image-path datasets --feature-extraction-cnn resnet101 --trained-model-dir trained_models  --partial-occlusion True --trained-model-fn hom_occ

# #####################################

# Retraining models with occlusion 

# echo "Reteraining TPS 8P occlusion"
# python3 train_strong.py --geometric-model tps --num-epochs 20 --training-dataset streetview  --dataset-csv-path training_data/streetview-random --dataset-image-path datasets --feature-extraction-cnn resnet101 --trained-model-dir trained_models --reg-grid False --trained-model-fn tps_8p_no_occ_occ --pretrained-model trained_models/tps_8p_no_occ/best_tps_8p_no_occ_tps_grid_lossresnet101.pth.tar --x-axis-coords -1.0 -1.0 -1.0 0.0 0.0  1.0 1.0 1.0 --y-axis-coords -1.0 0.0 1.0 -1.0 1.0 -1.0 0.0 1.0
# echo "Reteraining TPS 4P occlusion"
# python3 train_strong.py --geometric-model tps --num-epochs 20 --training-dataset streetview  --dataset-csv-path training_data/streetview-random --dataset-image-path datasets --feature-extraction-cnn resnet101 --trained-model-dir trained_models --reg-grid False --trained-model-fn tps_4p_no_occ_occ --pretrained-model trained_models/tps_4p_no_occ/best_tps_4p_no_occ_tps_grid_lossresnet101.pth.tar --x-axis-coords -1.0 -1.0 1.0 1.0 --y-axis-coords -1.0 1.0 -1.0 1.0
# echo "Retraining TPS 9P occlusion"
# python3 train_strong.py --geometric-model tps --num-epochs 20 --training-dataset streetview --dataset-csv-path training_data/streetview-random --dataset-image-path datasets --feature-extraction-cnn resnet101 --trained-model-dir trained_models --reg-grid True --trained-model-fn tps_9p_no_occ_occ --pretrained-model trained_models/tps_9p_no_occ/best_tps_9p_no_occ_tps_grid_lossresnet101.pth.tar
# echo "Retraining Affine occlusion"
# python3 train_strong.py --geometric-model affine --num-epochs 20 --training-dataset streetview  --dataset-csv-path training_data/streetview-random --dataset-image-path datasets --feature-extraction-cnn resnet101 --trained-model-dir trained_models --trained-model-fn affine_no_occ_occ --pretrained-model trained_models/affine_no_occ/best_affine_no_occ_tps_grid_lossresnet101.pth.tar
# echo "Retraining Hom no occlusion"
# python3 train_strong.py --geometric-model hom --num-epochs 20 --training-dataset streetview --dataset-csv-path training_data/streetview-random --dataset-image-path datasets --feature-extraction-cnn resnet101 --trained-model-dir trained_models --trained-model-fn hom_no_occ_occ --pretrained-model trained_models/hom_no_occ/best_hom_no_occ_tps_grid_lossresnet101.pth.tar


# # # Retraining models without occlusion

# echo "Retraining TPS 8P  no occlusion"
# python3 train_strong.py --geometric-model tps --num-epochs 20 --training-dataset streetview --log_interval 50 --dataset-csv-path training_data/streetview-random --dataset-image-path datasets --feature-extraction-cnn resnet101 --trained-model-dir trained_models --reg-grid False --partial-occlusion False --trained-model-fn tps_8p_occ_no_occ --pretrained-model trained_models/tps_8p_occ/best_tps_8p_occ_tps_grid_lossresnet101.pth.tar --x-axis-coords -1.0 -1.0 -1.0 0.0 0.0  1.0 1.0 1.0 --y-axis-coords -1.0 0.0 1.0 -1.0 1.0 -1.0 0.0 1.0
# echo "Retraining TPS 4P  no occlusion"
# python3 train_strong.py --geometric-model tps --num-epochs 20 --training-dataset streetview --log_interval 50 --dataset-csv-path training_data/streetview-random --dataset-image-path datasets --feature-extraction-cnn resnet101 --trained-model-dir trained_models --reg-grid False --partial-occlusion False --trained-model-fn tps_4p_occ_no_occ --pretrained-model trained_models/tps_4p_occ/best_tps_4p_occ_tps_grid_lossresnet101.pth.tar --x-axis-coords -1.0 -1.0 1.0 1.0 --y-axis-coords -1.0 1.0 -1.0 1.0
# echo "Retraining TPS 9P no occlusion"
# python3 train_strong.py --geometric-model tps --num-epochs 20 --training-dataset streetview --log_interval 50 --dataset-csv-path training_data/streetview-random --dataset-image-path datasets --feature-extraction-cnn resnet101 --trained-model-dir trained_models --reg-grid True  --partial-occlusion False --trained-model-fn tps_9p_occ_no_occ --pretrained-model trained_models/tps_9p_occ/best_tps_9p_occ_tps_grid_lossresnet101.pth.tar
# echo "Retraining Affine no occlusion"
# python3 train_strong.py --geometric-model affine --num-epochs 20 --training-dataset streetview --log_interval 50 --dataset-csv-path training_data/streetview-random --dataset-image-path datasets --feature-extraction-cnn resnet101 --trained-model-dir trained_models  --partial-occlusion False --trained-model-fn affine_occ_no_occ --pretrained-model trained_models/affine_occ/best_affine_occ_affine_grid_lossresnet101.pth.tar
# echo "Retraining Hom no occlusion"
# python3 train_strong.py --geometric-model hom --num-epochs 20 --training-dataset streetview --log_interval 50 --dataset-csv-path training_data/streetview-random --dataset-image-path datasets --feature-extraction-cnn resnet101 --trained-model-dir trained_models  --partial-occlusion False --trained-model-fn hom_occ_no_occ --pretrained-model trained_models/hom_occ/best_hom_occ_hom_grid_lossresnet101.pth.tar