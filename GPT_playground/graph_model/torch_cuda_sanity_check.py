import torch

# Check if CUDA is available
if torch.cuda.is_available(): 
    # You can optionally print the CUDA device name
    print(torch.cuda.get_device_name(0))

    # Create a tensor and move it to GPU memory
    a = torch.ones(3, 3)
    a = a.to('cuda')

    # Perform a simple operation
    b = a + a
    print(b)

    # Move the tensor from GPU memory to system memory
    b = b.to('cpu')
    print(b)

else:
    print('CUDA is not available.')
