import  torch

if __name__ == "__main__":

    model_path = './optimizer.pt'
    model = torch.load(model_path, map_location=torch.device('cpu'))
    torch.save(model,'./adapter_model.bin')