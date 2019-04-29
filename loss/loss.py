
class Loss(object):
    def __init__(self, config):
        self.loss_type = config.loss_type
        self.gamma = config.gamma if hasattr(config, "gamma") else 2.0
        self.alpha = config.alpha if hasattr(config, "alpha") else 0.25
        self.lamda = config.lamda if hasattr(config, "lamda") else 0.05
        self.margin = config.binary_margin if hasattr(config, "binary_margin") else 0.8

    def getLoss(self, logits, labels):
        if self.loss_type == "focal_loss":
            print("loss_type: focal_loss")
            return self.focalLoss(logits, labels, self.gamma, self.alpha)
        elif self.loss_type == "uni_loss":
            print("loss_type: uni_loss")
            return self.uniLoss(logits, labels, self.lamda)
        elif self.loss_type == "hard_cross_entropy":
            print("loss_type: hard_cross_entropy")
            return self.hardCrossEntropy(logits, labels, self.margin)
        else:
            print("loss_type: cross_entropy")
            return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels,tf.float32)))

    def focalLoss(self, logits, labels, gamma=2.0, alpha=0.25, epsilon=1.e-8):
        logits = tf.cast(logits, tf.float32)
        model_out = tf.add(logits, epsilon)
        ce = tf.multiply(tf.cast(labels, tf.float32), -tf.log(model_out))
        weights = tf.multiply(tf.cast(labels, tf.float32), tf.pow(tf.subtract(1.0, model_out), gamma))

        focal_loss = tf.reduce_mean(tf.multiply(alpha, tf.multiply(weights, ce)))
        return focal_loss

    def uniLoss(self, logits, labels, lamda=0.05, loss_func="cross_entropy"):
        if loss_func == "focal_loss":
            model_loss = self.focalLoss(logits, labels)
        else:
            model_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.cast(labels, tf.float32)))
        num_classes = tf.cast(logits.get_shape()[-1], tf.int32)
        uni_labels = tf.fill([1, num_classes], 1.0 / tf.cast(num_classes, tf.float32))
        uni_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=uni_labels))
        loss = lamda * uni_loss + (1.0 - lamda) * model_loss
        return loss

    def hardCrossEntropy(self, logits, labels, margin=0.8, eps=1e-8): #for binary classfication
        theta = lambda t: (tf.sign(t) + 1.0) / 2.0
        margin = tf.cast(margin, tf.float32)
        labels = tf.cast(labels, tf.float32)
        lamda = (1.0 - theta(labels - margin) * theta(logits - margin) - theta(1.0 - margin - labels) * theta(1.0 - margin - logits))
        hard_loss = - lamda * (labels * tf.log(logits + eps) + (1 - labels) * tf.log(1 - logits + eps))
        return tf.reduce_mean(hard_loss)


