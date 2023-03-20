arch_list=(resnet20 resnet32 resnet44)

for arch in ${arch_list[@]};
do
  model_args="dataset=simplemnist_model=${arch}_epoch=50_bs=128_lr=0.01_logspace=1_seed=0"
  segment_version="version-2"
  python3 finetune-baseline.py --model-args=$model_args --segment-version=$segment_version \
    --selected-dim=gt-log-odds --baseline-init=zero --baseline-lb=0.0 --baseline-ub=0.2 \
    --finetune-lr=1e-3 --finetune-max-iter=50 --calc-bs=32 --gpu-id=3
done