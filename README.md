# Notes

Encode uses v ⊗ v instead of v for testing purposes.

Good strategy is to use the "default" loss function from https://arxiv.org/abs/1804.00633 at first. It's implemented in *target* function. After finding good parameters - use loss function with additional sigmoid. It's implemented in *target2* function. It finds much better parameters. 

