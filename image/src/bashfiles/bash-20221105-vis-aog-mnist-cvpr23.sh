#arch_list=(resnet20 resnet32 lenet resnet44 vgg13_bn vgg16_bn)
#arch_list=(resnet20 resnet32 resnet44 vgg13_bn vgg16_bn)
#arch_list=(resnet20 resnet32 lenet vgg16_bn)
arch_list=(resnet20)

for arch in ${arch_list[@]};
do
  model_args="dataset=simplemnist_model=${arch}_epoch=50_bs=128_lr=0.01_logspace=1_seed=0"
  segment_args="manual_segment_version-2"
  harsanyi_args="dim=gt-log-odds_baseline-init=zero_lb=0.0_ub=0.1_lr=0.001_niter=50"
  python3 generate-aog.py --model-args=$model_args --segment-args=$segment_args --harsanyi-args=$harsanyi_args
done


