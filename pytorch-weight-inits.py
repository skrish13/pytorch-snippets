def _initialize_weights(self):
    '''
    ResNet style
    '''
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
          n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
          m.weight.data.normal_(0, math.sqrt(2. / n))
      elif isinstance(m, nn.BatchNorm2d):
          m.weight.data.fill_(1)
          m.bias.data.zero_()
 
def _initialize_weights(self):
    '''
    VGG style
    '''
	for m in self.modules():
		if isinstance(m, nn.Conv2d):
			n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
			m.weight.data.normal_(0, math.sqrt(2. / n))
			if m.bias is not None:
				m.bias.data.zero_()
		elif isinstance(m, nn.BatchNorm2d):
			m.weight.data.fill_(1)
			m.bias.data.zero_()
		elif isinstance(m, nn.Linear):
			m.weight.data.normal_(0, 0.01)
			m.bias.data.zero_()

def _initialize_weights(self):
    '''
    Ex: Squeezenet
    '''
	for m in self.modules():
		if isinstance(m, nn.Conv2d):
			if m is final_conv:
				init.normal(m.weight.data, mean=0.0, std=0.01)
			else:
				init.kaiming_uniform(m.weight.data)
			if m.bias is not None:
				m.bias.data.zero_()
