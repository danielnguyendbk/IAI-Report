from dataset import DataPipeline
from ood_generator import OODConfig, OODGenerator

pipeline = DataPipeline("data/Arrhythmia_raw_clean.csv")
bundle = pipeline.run()

generator = OODGenerator(OODConfig(method="shuffle"))
X_ood = generator.generate(bundle.X_test)

print("Train:", bundle.X_train.shape)
print("Val:", bundle.X_val.shape)
print("Test ID:", bundle.X_test.shape)
print("Test OOD:", X_ood.shape)