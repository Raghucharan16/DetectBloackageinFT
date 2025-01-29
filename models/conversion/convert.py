import torch
import tensorrt as trt

def export_onnx():
    model = MedicalX3D()
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    
    dummy_input = torch.randn(1, 1, 16, 224, 224)
    torch.onnx.export(
        model, dummy_input, "medical_x3d.onnx",
        input_names=["input"], output_names=["output"],
        dynamic_axes={'input': {0: 'batch'}, 'output': {0: 'batch'}}
    )

def build_trt_engine():
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    with open("medical_x3d.onnx", "rb") as f:
        parser.parse(f.read())
        
    config = builder.create_builder_config()
    config.set_flag(trt.BuilderFlag.FP16)
    config.max_workspace_size = 1 << 30
    
    engine = builder.build_engine(network, config)
    with open("medical_x3d.engine", "wb") as f:
        f.write(engine.serialize())
