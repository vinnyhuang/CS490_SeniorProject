"""Code adapted from https://github.com/experiencor/deep-viz-keras"""
from saliency import SaliencyMask
import numpy as np
import keras.backend as K
# from keras.layers import Input, Deconvolution2  #Keras 1 Adaptation
from keras.layers import Input, Conv2DTranspose
from keras.models import Model
from keras.initializers import Ones, Zeros

class VisualBackprop(SaliencyMask):
    """A SaliencyMask class that computes saliency masks with VisualBackprop (https://arxiv.org/abs/1611.05418).
    """

    def __init__(self, model, output_index=0):
        """Constructs a VisualProp SaliencyMask."""
        inps = [model.input, K.learning_phase()]           # input placeholder
        outs = [layer.output for layer in model.layers]    # all layer outputs
        self.forward_pass = K.function(inps, outs)         # evaluation function
        
        self.model = model

    def get_mask(self, input_image):
        """Returns a VisualBackprop mask."""
        x_value = np.expand_dims(input_image, axis=0)
        
        visual_bpr = None
        layer_outs = self.forward_pass([x_value, 0])

        for i in range(0, len(self.model.layers)):
            print (i, str(type(self.model.layers[i])), self.model.layers[i].input_shape, self.model.layers[i].output_shape)

        for i in range(len(self.model.layers)-1, -1, -1):
            print (i, self.model.layers[i], np.shape(visual_bpr))
            if 'Conv2D' in str(type(self.model.layers[i])):
                layer = np.mean(layer_outs[i], axis=3, keepdims=True)
                layer = layer - np.min(layer)
                layer = layer/(np.max(layer)-np.min(layer)+1e-6)

                if visual_bpr is not None:
                    if visual_bpr.shape != layer.shape:
                        visual_bpr = self._deconv(visual_bpr, i)
                    visual_bpr = visual_bpr * layer
                else:
                    visual_bpr = layer

        visual_bpr = self._deconv(visual_bpr, 0)
        return visual_bpr[0]
    
    def _deconv(self, feature_map, i):
        """The deconvolution operation to upsample the average feature map downstream"""
        x = Input(shape=(None, None, 1))
        if i >= 3:
            k1 = 3
            k2 = 3
            s = 1
        elif i == 2:
            k1 = 6
            k2 = 5
            s = 2
        elif i == 1:
            k1 = 5
            k2 = 6
            s = 2
        elif i <= 0:
            k1 = 6
            k2 = 6
            s = 2
        
        y = Conv2DTranspose(filters=1, 
                            kernel_size=(k1,k2), 
                            padding='valid',
                            bias_initializer=Zeros(),
                            kernel_initializer=Ones(),
                            strides=(s,s))(x)


        deconv_model = Model(inputs=[x], outputs=[y])

        inps = [deconv_model.input, K.learning_phase()]   # input placeholder                                
        outs = [deconv_model.layers[-1].output]           # output placeholder
        deconv_func = K.function(inps, outs)              # evaluation function
        
        return deconv_func([feature_map, 0])[0]
