# AOG code (Tabular data)

## Requirements

TBD

## Toy demo

A simple example to calculate Harsanyi dividends.

~~~shell
python demo.py
~~~

The visualized AOG will be saved at `./AOG_code/tabular/tmp`.

## Usage

- To train the model,

~~~shell
python train_model.py --device=2 --dataset=commercial --arch=mlp5
~~~

- To train baseline values,

~~~shell
python finetune_baseline.py --device=2 --dataset=commercial --arch=mlp5
~~~

- To calculate Harsanyi dividends,

~~~shell
python eval_model-remove-noisy.py --device=2 --dataset=commercial --arch=mlp5
~~~

- To visualize the AOG,
- 
~~~shell
python visualize-AOG-remove-noisy.py --device=2 --dataset=commercial --arch=mlp5
~~~

- Note

For all the above, please run `python ${filename}.py --help` for more information on arguments.

## Dataset location

122.51.159.29:10035 (22001)

- commercial: `/data1/limingjie/data/tabular/commercial`
- commercial (raw): `/data1/limingjie/data/tabular/TV_News_Channel_Commercial_Detection_Dataset`
- census: `/data1/limingjie/data/tabular/census`
- census (raw): `/data1/limingjie/data/tabular/Census-Dataset-Raw`
- bike: `/data1/limingjie/data/tabular/bike`
- bike (raw): `/data1/limingjie/data/tabular/Bike-Sharing-Dataset`