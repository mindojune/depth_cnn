"""
Created on Fri Dec 15 19:57:34 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
import os
import numpy as np
import cv2

import torch
from torch import nn
from torch.autograd import Variable
# from torch.autograd.gradcheck import zero_gradients  # See processed_image.grad = None

from misc_functions import preprocess_image, recreate_image, get_params
import torch.nn.functional

class FastGradientSignUntargeted():
    """
        Fast gradient sign untargeted adversarial attack, minimizes the initial class activation
        with iterative grad sign updates
    """
    def __init__(self, model, alpha):
        self.model = model
        self.model.eval()
        # Movement multiplier per iteration
        self.alpha = alpha
        # Create the folder to export images if not exists
        if not os.path.exists('../generated'):
            os.makedirs('../generated')

    def generate(self, original_image, im_label):
        # I honestly dont know a better way to create a variable with specific value
        im_label_as_var = Variable(torch.from_numpy(np.asarray([im_label])))
        im_label_as_var = im_label_as_var.long()
        im_label_as_var = torch.max(im_label_as_var, 1)[1]
        
        # Define loss functions
        ce_loss = nn.CrossEntropyLoss()
        # Process image
        processed_image = preprocess_image(original_image)
        # Start iteration
        for i in range(10):
            #print('Iteration:', str(i))
            # zero_gradients(x)
            # Zero out previous gradients
            # Can also use zero_gradients(x)
            processed_image.grad = None
            # Forward pass
            out = self.model(processed_image)
            # Calculate CE loss
            
            # no need to softmax out
            #out = torch.nn.functional.softmax(Variable(out, requires_grad=True), dim=1).data

            im_label = im_label_as_var

            pred_loss = ce_loss(out, im_label_as_var)
            #print(pred_loss)
            # Do backward pass
            pred_loss.backward()
            # Create Noise
            # Here, processed_image.grad.data is also the same thing is the backward gradient from
            # the first layer, can use that with hooks as well
            adv_noise = self.alpha * torch.sign(processed_image.grad.data)
            # Add Noise to processed image
            processed_image.data = processed_image.data + adv_noise

            # Confirming if the image is indeed adversarial with added noise
            # This is necessary (for some cases) because when we recreate image
            # the values become integers between 1 and 255 and sometimes the adversariality
            # is lost in the recreation process

            # Generate confirmation image
            recreated_image = recreate_image(processed_image)
            # Process confirmation image
            prep_confirmation_image = preprocess_image(recreated_image)
            # Forward pass
            confirmation_out = self.model(prep_confirmation_image)
            # Get prediction
            _, confirmation_prediction = confirmation_out.data.max(1)
            #print(confirmation_prediction)
            # Get Probability
            confirmation_confidence = \
                nn.functional.softmax(confirmation_out)[0][confirmation_prediction].data.numpy()[0]
            # Convert tensor to int
            confirmation_prediction = confirmation_prediction.numpy()[0]
            # Check if the prediction is different than the original
            if confirmation_prediction != im_label:
                #print('Original image was predicted as:', im_label,
                #      'with adversarial noise converted to:', confirmation_prediction,
                #      'and predicted with confidence of:', confirmation_confidence)
                # Create the image for noise as: Original image - generated image
                noise_image = original_image - recreated_image
                # cv2.imwrite('../generated/untargeted_adv_noise_from_' + str(im_label) + '_to_' +
                #             str(confirmation_prediction) + '.jpg', noise_image)
                # # Write image
                # cv2.imwrite('../generated/untargeted_adv_img_from_' + str(im_label) + '_to_' +
                #             str(confirmation_prediction) + '.jpg', recreated_image)
                return recreated_image, 1

        return None, 0


if __name__ == '__main__':
    target_example = 2  # Eel
    (original_image, prep_img, target_class, _, pretrained_model) =\
        get_params(target_example)

    FGS_untargeted = FastGradientSignUntargeted(pretrained_model, 0.01)
    FGS_untargeted.generate(original_image, target_class)