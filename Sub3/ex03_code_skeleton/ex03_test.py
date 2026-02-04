from ex03_model import ShallowCNN
from ex03_main import run_generation, run_evaluation, parse_args, run_ood_analysis
import torch
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

CHECKPOINT_PATH = "Sub3/JEM/lightning_logs/version_1/checkpoints/last_epoch=115-step=40948.ckpt"

def test_shallow_cnn_forward():
    model = ShallowCNN(hidden_features=16, num_classes=10)
    batch_size = 8
    input_tensor = torch.randn(batch_size, 1, 64, 64)

    # Test unconditional forward pass
    energy_uncond = model(input_tensor)
    assert energy_uncond.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {energy_uncond.shape}"

    # Test conditional forward pass
    labels = torch.randint(0, 10, (batch_size,))
    energy_cond = model(input_tensor, labels)
    assert energy_cond.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {energy_cond.shape}"

def test_shallow_cnn_get_logits():
    model = ShallowCNN(hidden_features=16, num_classes=10)
    batch_size = 26
    input_tensor = torch.randn(batch_size, 1, 56, 56)
    logits = model.get_logits(input_tensor)
    assert logits.shape == (batch_size, 10), f"Expected shape {(batch_size, 10)}, got {logits.shape}"
    assert torch.isnan(logits).sum() == 0, "Logits contain NaN values"

def test_generate_images():
    # checkpoint_path = "saved_models/lightning_logs/version_61/checkpoints/last_epoch=1-step=706.ckpt"
    checkpoint_path = CHECKPOINT_PATH
    num_steps = 120
    args = {'num_steps': num_steps}
    run_generation(args, checkpoint_path, conditional=False)

def test_pytorch_grad():
    inp_imgs = torch.normal(0, 0.01, [32,1,56,56], device=device) * 2.0 - 1.0
    model = ShallowCNN(hidden_features=16, num_classes=10).to(device)
    # with torch.enable_grad():
        # inp_imgs.requires_grad = True
        # model.train(True)
        # for param in model.cnn_layers.parameters():
        #     param.requires_grad = True
        # for param in model.fc_layers.parameters():
        #     param.requires_grad = True

    out = inp_imgs
    for i, layer in enumerate(model.cnn_layers):
        out = layer(out)
        print(f"Layer {i} ({layer.__class__.__name__}): Out grad_fn is {out.grad_fn}")
        if out.grad_fn is None:
            print("!!! THE CHAIN BROKE AT THE LAYER ABOVE !!!")
            break



def test_evaluation():
    checkpoint_path = CHECKPOINT_PATH
    args = parse_args()
    run_evaluation(args, checkpoint_path)

def test_ood_analysis():
    checkpoint_path = CHECKPOINT_PATH
    args = parse_args()
    run_ood_analysis(args, checkpoint_path)


if __name__ == "__main__":
    test_generate_images()