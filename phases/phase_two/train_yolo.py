from ultralytics import YOLO

model = YOLO('yolo26n.pt')

#train the model on the dataset
results = model.train(data='datasets/dataset/data.yaml',
                      epochs=100,
                      imgsz=640,
                      patience=10,
                      batch=16,
                      device='cpu',
                      project='phases/phase_two',
                      name='yolo26n_trained',
                      exist_ok=True,
                      optimizer='auto',
                      seed=42,)

metrics = model.val(data='datasets/dataset/data.yaml',
                    split='val')

#save the model to a file
model.save('phases/phase_two/yolo26n_trained.pt')

print(metrics.box.map50)
print(metrics.box.mr)
print(metrics.box.map)