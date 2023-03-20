# AOG code (NLP data)

## Requirements

TBD

## Toy demo

A simple example to calculate Harsanyi dividends. Please see `./AOG_code/tabular` folder.

~~~shell
python demo.py
~~~

The visualized AOG will be saved at `./AOG_code/tabular/tmp`.

## Usage

- To train the model,

~~~shell
python train-nlp-model.py --gpu_id=2 --dataset=CoLA --arch=cnn --epoch=5
~~~

- To train baseline values,

~~~shell
python finetune-baseline.py --gpu_id=2 --dataset=CoLA --arch=cnn --epoch=5
~~~

- To calculate Harsanyi dividends,

~~~shell
python eval-nlp-model-remove-noisy.py --gpu_id=2 --dataset=CoLA --arch=cnn --epoch=5
~~~

- To visualize the AOG,
- 
~~~shell
python visualize-AOG-remove-noisy.py --gpu_id=2 --dataset=CoLA --arch=cnn --epoch=5
~~~

- Note

For all the above, please run `python ${filename}.py --help` for more information on arguments.

## Data location

122.51.159.29:10035 (22001)

- CoLA: `/data1/limingjie/data/NLP/CoLA`
- CoLA (raw): `/data1/limingjie/data/NLP/CoLA_raw`
- SST-2: `/data1/limingjie/data/NLP/SST-2`
- SST-2 (raw): `/data1/limingjie/data/NLP/SST-2_raw`