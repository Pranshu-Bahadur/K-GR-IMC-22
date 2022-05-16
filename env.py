exec(open('utils.py', 'r').read())
exec(open('run_configs/experiment.bat', 'r').read())

t_dir = '/home/pranshu-bahadur/Downloads/datasets/image-matching-challenge-2022/train'

training_dataset = TrainIMC22Dataset(t_dir)

splits = split_dataset(training_dataset, 0.8)

training_args = TrainingArguments(**training_args)




train_args = {
        'model': IMC22Model('resnet18'),
        'train_dataset': splits[0],
        'eval_dataset': splits[1],
        'args': training_args
}




class IMC22Trainer(Trainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        pass

    def compute_loss(self, model, inputs, return_outputs=True):
       """
       Inputs: Batch x Tuple[List[Tensor], Tensor]
       """
       x, y = inputs
       outputs = model(x)
       loss = self.loss_fct(outputs.view(y.size(0),-1), y)
       return (loss, outputs)



class IMC22Model(torch.nn.Module):
    def __init__(self, model_name, **kwargs):
        self.model1 = create_model(model_name, num_classes=1, pretrained=True, in_chans=3)
        self.model1.head.fc = torch.nn.Linear(self.model1.classifier.in_features, 1)
        self.model2 = create_model(model_name, num_classes=1, pretrained=True, in_chans=3)
        self.model2.head.fc = torch.nn.Linear(self.model2.classifier.in_features, 1)

    def forward(self, x : List[Tensor]):
        return self.model1(x) + self.model2






print(IMC22Trainer(**train_args).model)
