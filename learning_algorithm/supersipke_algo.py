import torch
from norse.torch.functional.heaviside import heaviside  

# Define a custom autograd function for the SuperSpike surrogate gradient.
class SuperSpike(torch.autograd.Function):
    r"""
    SuperSpike surrogate gradient as described in Section 3.3.2 of

    

    This class implements the forward and backward methods needed for custom autograd operations.

    This is implemented when method is set to "super" in the lifrecurrent layers

    """

    @staticmethod
    def forward(input_tensor: torch.Tensor, alpha: float) -> torch.Tensor:
        """
        Forward pass for the SuperSpike function.

        Parameters:
        input_tensor (torch.Tensor): The input tensor.
        alpha (float): Scaling factor for the backward pass.

        Returns:
        torch.Tensor: The output tensor after applying the Heaviside function.
        """
        return heaviside(input_tensor)  # Apply the Heaviside step function to the input tensor.

    @staticmethod
    def setup_context(ctx, inputs, output):
        """
        Setup context for backpropagation.

        Parameters:
        ctx: Context object to store information for the backward pass.
        inputs: Tuple containing the input tensor and alpha.
        output: The output tensor from the forward pass.

        This method saves the input tensor and alpha to the context object for use in the backward pass.
        """
        input_tensor, alpha = inputs
        # Save alpha for backward pass
        ctx.alpha = alpha  
        # Save  input tensor for backpropagation.
        ctx.save_for_backward(input_tensor)  

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass for the SuperSpike function.

        Parameters:
        ctx: Context object containing saved information from the forward pass.
        grad_output: Gradient of the loss with respect to the output of the forward pass.

        Returns:
        torch.Tensor: Gradient of the loss with respect to the input of the forward pass.
        """
        (inp,) = ctx.saved_tensors  # Retrieve the saved input tensor.
        alpha = ctx.alpha  # Retrieve the saved alpha value.
        grad = None
        # Check if the gradient with respect to the input is require
        if ctx.needs_input_grad[0]:  
             # Calculate and retuen the gradient
            
            grad = grad_output / (alpha * torch.abs(inp) + 1.0).pow(2) 
        return grad, None  

# Function to apply the SuperSpike surrogate gradient.
def super_fn(x: torch.Tensor, alpha: float = 100.0) -> torch.Tensor:
    """
    Apply the SuperSpike function to the input tensor.

    Parameters:
    x (torch.Tensor): The input tensor.
    alpha (float): Scaling factor for the backward pass.

    Returns:
    torch.Tensor: The output tensor after applying the SuperSpike function.
    """
    return SuperSpike.apply(x, alpha) 
