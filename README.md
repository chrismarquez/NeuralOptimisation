
## Neural Optimisation Software Archive

This is the software archive related to the work presented in the paper *Behavioural Patterns in Neural Surrogate
Models subject to Optimisation Methods*.

### Dependencies

To clone the project to include the OMLT submodule, use the command:

```bash
    git clone --recurse-submodules https://github.com/chrismarquez/NeuralOptimisation
```

To install the required dependencies, use the following command:

```bash
    pip install -r requirements.txt
```

Additionally, clone the OMLT repository at the root f

### Run the Experiment Suite

The main program for the experimental suite is executed with the following command:

```bash
    python3 main.py
```

The following flags are available for use. Some of them are required and others are optional:
- `--experiment`: Used to define this run’s experiment ID. [Required]
- `--epochs`: Used to designate the number of training epochs for this experiment. [Required]
- `--type`: Type of Nueral Model to use. Can be either `Feedforward` or `Convolutional`. [Required]
- `--l1reg`: Used to define the L1 Regularization constant’s value during neural
training. [Optional]
- `--test`: Used to run an experiment in test mode. Only the first 10 models are
trained and optimised. [Optional]
- `--debug`: Used to run an experiment in debug mode. Debug data about cluster
job statuses is displayed. [Optional]
- `--local`: Used to run the experiment’s tasks in local hardware instead of the
institutional clusters [Optional]

An example of the command used to run a regularized set of experiments would be the following:

```bash
    python3 main.py --experiment batch-1-reg --type Feedforward --epoch 200 --l1reg 0.0005
```

## Restore Database Records

This repository includes a database dump with previously trained experiments. In order to
restore this data, a working installation of MongoDB is necessary. To restore the database
dump, the following command can be used:

```bash
    mongorestore --db=NeuralOptimisation --collection=NeuralModel NeuralModel.bson
```
In this case, the restored collection name will be `NeuralModel` and the data will
be restored from the `NeuralModel.bson` dump file.