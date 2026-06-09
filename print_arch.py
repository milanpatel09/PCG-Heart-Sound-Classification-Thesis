import torch
from torchinfo import summary

# Import your custom model class from your models.py file
from src.models import AudioResNet 

def main():
    print("Loading AudioResNet-18 Architecture...\n")
    
    # 1. Instantiate your model exactly as you use it in your pipeline
    # We set pretrained=False here just to load the architecture quickly
    model = AudioResNet(architecture='resnet18', num_classes=2, pretrained=False)
    
    # 2. Define the exact shape of your time-frequency inputs
    # Based on your dataset.py: (Batch Size, Channels, Height, Width)
    # You are feeding 3-channel (RGB) 224x224 images into the ResNet
    batch_size = 1
    input_shape = (batch_size, 3, 224, 224)
    
    # 3. Print the beautiful summary table
    print("=" * 90)
    print(f"{'AUDIO RESNET-18 (MODIFIED FOR PCG) ARCHITECTURE':^90}")
    print("=" * 90)
    
    summary(
        model, 
        input_size=input_shape,
        col_names=["input_size", "output_size", "num_params", "kernel_size"],
        col_width=20,
        row_settings=["var_names"]
    )

if __name__ == "__main__":
    main()