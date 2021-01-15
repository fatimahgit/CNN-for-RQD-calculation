import torch
import torch.nn as nn
from torchvision.transforms import transforms
from torch.autograd import Variable
from PIL import Image
from resizeimage import resizeimage

# define the model for loading
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=7, stride=2,  padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size= 3, stride=1, padding =1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7* 7, 1000),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(1000, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 128 * 7 * 7)
        x = self.classifier(x)
        return x

model_struc = Net()


def load_model(model_path,class_to_idx_path,model=model_struc, device= torch.device('cpu')):
    class_to_idx = torch.load(class_to_idx_path, map_location=device)['class_to_idx']
    params = torch.load(model_path, map_location='cpu')
    model.load_state_dict(params)
    model.to(device)
    model.eval()
    return (model,class_to_idx)


def predict_image(image,model, class_to_idx):
    #a function to predict the image using CNN in each sliding window
    image = Image.fromarray(image)
    image = resizeimage.resize_cover(image, [225, 225])  # resize the image, no crop
    transformation = transforms.Compose([transforms.ToTensor(),
                                         transforms.Normalize((0.546, 0.479, 0.460), (0.189, 0.189, 0.185))])
    image_tensor = transformation(image).float()
    image_tensor = image_tensor.unsqueeze_(0)
    if torch.cuda.is_available():
        image_tensor.cuda()
        input = Variable(image_tensor).cuda()
    else:
        input = Variable(image_tensor)
    output = model(input)
    max_value, max_index = torch.max(output, 1)
    class_name = list(class_to_idx.keys())[list(class_to_idx.values()).index(max_index.item())]
    return (class_name)