# Explanation of network structure and training approach

Track 1 sample data
https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip



Tried with (16, 32) small network starting 1x1 convolution.

Car drives straight off in the first curve.
Tried removing all the straight driving and found improved performance.
Experiemnt with 10% of the straight driving in the data set. Worse performance. Validation error going up.



- NVIDIA paper [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)
- Vivek Yadav [blog post](https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9#.1nbgoagsm)
- comma.ai [steering model code](https://github.com/commaai/research/blob/master/train_steering_model.py)
- Subodh Malgonde for [blog post](https://medium.com/@subodh.malgonde/teaching-a-car-to-mimic-your-driving-behaviour-c1f0ae543686#.ndr91eurb) and [code](https://github.com/subodh-malgonde/behavioral-cloning)
- Paul Heraty [forum post](https://carnd-forums.udacity.com/cq/viewquestion.action?id=26214464&questionTitle=behavioral-cloning-cheatsheet)