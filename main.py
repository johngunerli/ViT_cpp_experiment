import torch
from transformers import ViTModel


class ViTEmbeddingExtractor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)
        return output.last_hidden_state[:, 0, :]  # extract embedding


model = ViTModel.from_pretrained("google/vit-base-patch16-224")
wrapped_model = ViTEmbeddingExtractor(model)
wrapped_model.eval()

# Dummy input tensor (batch size 1, 3 color channels, 224x224 image)
dummy_input = torch.rand(1, 3, 224, 224)

scripted_model = torch.jit.trace(wrapped_model, dummy_input)
scripted_model.save("vit_embedding.pt")

print("Model saved successfully!")
