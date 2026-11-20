from guidewire_seg.experiment import image,dice,evaluate
def test_image():
 x,m=image(2,64);assert x.shape==m.shape==(64,64) and m.any()
def test_metrics():assert all(0<=v["mean_dice"]<=1 for v in evaluate(3).values())

