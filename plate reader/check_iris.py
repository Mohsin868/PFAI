from ultralytics import YOLO

# Load the new specific plate model
model = YOLO('best.pt') 

# Convert to OpenVINO format
model.export(format='openvino')
print("✅ Conversion Complete! Your Intel Iris GPU is ready.")