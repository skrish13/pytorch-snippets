# Few tips for partial fine tuning

* Usually in most of the available docs/discuss.pytorch, transfer learning would just require changing the linear layer and fine-tuning would involve fine-tuning all the layers in the network.
* But one would also really like to freeze few layers and fine-tune rest, `torchvision.models` doesn't really allow you to do that in an easy way.
* But there are 2 ways you could do it
	* Reconstruct the Sequential using `list(model.children())`, make changes appropriately to it and form a new Sequential from that list.
	* Manually calculate the indices of layers to be fine-tuned, and don't make `requires_grad=False` for them.
* After that, pass only parameters which you'd want to train (requires_grad=True)
	* Ex: `(filter(lambda p: p.requires_grad, net.parameters())`
