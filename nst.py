import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.applications import vgg19
from tensorflow.keras.preprocessing.image import load_img, save_img, img_to_array, array_to_img
from tensorflow.keras import backend as K


class NST:
    def __init__(self, content_image_path, style_image_path, final_image_path):
        self.content_img_path = content_image_path
        self.style_img_path = style_image_path
        self.content_layers = ['block5_conv2']
        self.style_layers = ['block1_conv1', 'block2_conv1',
                             'block3_conv1', 'block4_conv1', 'block5_conv1']
        self.model = self._get_model()
        width, height = load_img(self.content_img_path).size
        self.width = 400
        self.height = int(width * self.width / height)
        self.channels = 3
        self.style_weight = 0.2
        self.num_content_layers = len(self.content_layers)
        self.num_style_layers = len(self.style_layers)
        self.alpha = 1e3
        self.beta = 1e7
        self.total_variation_weight = 30
        self.model_output = None
        self.epochs = 100
        self.no_of_steps = 40
        self.learning_rate = 0.02
        self.beta_1 = 0.99
        self.epsilon = 1e-1
        self.opt = tf.optimizers.Adam(
            learning_rate=self.learning_rate, beta_1=self.beta_1, epsilon=self.epsilon)
        self.save_path = final_image_path

    def change_layers(self, content_layers, style_layers):
        self.content_layers = content_layers
        self.style_layers = style_layers

    def _get_model(self):
        vgg = vgg19.VGG19(include_top=False, weights='imagenet')
        vgg.trainable = False

        content_outputs = [vgg.get_layer(
            name).output for name in self.content_layers]
        style_outputs = [vgg.get_layer(
            name).output for name in self.style_layers]

        model_output = content_outputs + style_outputs

        model = tf.keras.Model([vgg.input], model_output)
        return model

    @staticmethod
    def _imshow(image, title=None):
        if len(image.shape) > 3:
            image = tf.squeeze(image, axis=0)
        plt.imshow(image)
        if title:
            plt.title(title)
        plt.show()

    def _preprocess_image(self, image_path):
        img = load_img(image_path, target_size=(self.width, self.height))
        img = img_to_array(img)
        img = img[np.newaxis, ...]
        img = img / 255.0
        return tf.convert_to_tensor(img)

    @staticmethod
    def _deprocess_image(tensor):
        tensor = tf.multiply(tensor, 255.0)
        if np.ndim(tensor) > 3:
            assert tensor.shape[0] == 1
            tensor = tensor[0]
        tensor = np.array(tensor, dtype=np.uint8)
        return array_to_img(tensor)

    @staticmethod
    def _process_for_vgg(img):
        img = img * 255.
        img = vgg19.preprocess_input(img)
        return img

    def _generate_input(self):
        G = tf.Variable(self._preprocess_image(self.content_img_path))
        return G

    @staticmethod
    def _gram_matrix(A):
        A = A[0]
        A = K.batch_flatten(K.permute_dimensions(A, (2, 0, 1)))
        result = K.dot(A, K.transpose(A))
        return result

    def _content_cost(self, content_image, generated_image):
        J = K.sum(K.square(content_image - generated_image)) / \
            (4 * self.height * self.width * self.channels)
        return J

    def _style_cost(self, style_image, generated_image):
        S_G = self._gram_matrix(style_image)
        G_G = self._gram_matrix(generated_image)
        J = K.sum(K.square(S_G - G_G)) / (4 * self.channels **
                                          2 * (self.width * self.height)**2)
        return self.style_weight * J

    def _total_cost(self, target_output):
        G_contents = target_output[:self.num_content_layers]
        G_styles = target_output[self.num_content_layers:]
        a_contents = self.model_output[:self.num_content_layers]
        a_styles = self.model_output[self.num_content_layers:]

        J_content = K.sum([self._content_cost(a_contents[i], G_contents[i])
                           for i in range(self.num_content_layers)])
        J_style = K.sum([self._style_cost(a_styles[i], G_styles[i])
                         for i in range(self.num_style_layers)])

        J = (self.alpha * J_content) + (self.beta * J_style)

        return J

    @staticmethod
    def clip_0_1(image):
        return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    def train_step(self, image):
        with tf.GradientTape() as tape:
            target_outputs = self.model(self._process_for_vgg(image))
            loss = self._total_cost(target_outputs)
            loss += self.total_variation_weight * \
                tf.image.total_variation(image)

        grad = tape.gradient(loss, image)
        self.opt.apply_gradients([(grad, image)])
        image.assign(self.clip_0_1(image))
        return loss

    def train(self):
        self.model = self._get_model()
        content_image = self._preprocess_image(self.content_img_path)
        style_image = self._preprocess_image(self.style_img_path)
        content_outputs = self.model(self._process_for_vgg(content_image))[
            :self.num_content_layers]
        style_outputs = self.model(self._process_for_vgg(style_image))[
            self.num_content_layers:]
        self.model_output = content_outputs + style_outputs
        generated_image = self._generate_input()
        step = 0
        for _ in range(self.epochs):
            for _ in range(self.no_of_steps):
                step += 1
                print(f"Step: {step}")
                loss = self.train_step(generated_image)
                print(f"Loss: {loss[0]}")
                if step % 10 == 0:
                    save_img(self.save_path,
                             self._deprocess_image(generated_image))


if __name__ == "__main__":
    content_image_path = 'data/stata.jpg'
    style_image_path = 'data/udnie.jpg'
    generated_image_path = 'data/generated.jpg'
    nst = NST(content_image_path, style_image_path, generated_image_path)
    nst.train()
