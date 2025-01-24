import tensorflow as tf
import math

class Trainer:
    def __init__(self, model, optimizer, loss_fn, lr_schedule, n_epochs, dataset='cifar10', early_stop_acc = 0.99):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.lr_schedule = lr_schedule
        self.n_epochs = n_epochs
        self.dataset = dataset
        self.initial_lr = optimizer.learning_rate.numpy()
        self.early_stop_acc = early_stop_acc  # Early stopping threshold
        self.stop_training = False  # Flag to stop training

    def compute_accuracy(self, y_true, y_pred, top_k=1):
        return tf.keras.metrics.sparse_top_k_categorical_accuracy(y_true, y_pred, k=top_k)

    def adjust_learning_rate(self, epoch, iteration, n_batches):
        if self.lr_schedule == "multistep":
            # Multi-step learning rate decay
            if self.dataset in ['cifar10', 'cifar100']:
                if epoch >= 0.75 * self.n_epochs:
                    decay = 2
                elif epoch >= 0.5 * self.n_epochs:
                    decay = 1
                else:
                    decay = 0
                lr = self.initial_lr * (0.1 ** decay)
            elif self.dataset == 'imagenet':
                decay = epoch // 30
                lr = self.initial_lr * (0.1 ** decay)
        elif self.lr_schedule == "cosine":
            # Cosine annealing learning rate
            T_total = self.n_epochs * n_batches
            T_cur = ((epoch - 1) % self.n_epochs) * n_batches + iteration
            lr = 0.5 * self.initial_lr * (1 + math.cos(math.pi * T_cur / T_total))
        else:
            raise ValueError("Unsupported LR schedule: Choose 'multistep' or 'cosine'.")

        self.optimizer.learning_rate.assign(lr)
        return lr  # Return the updated learning rate for printing

    def train_epoch(self, train_dataset, epoch):
        if self.stop_training:
            return 0, 0, 0  # Skip training if early stopping is triggered

        total_loss, total_top1, total_top5, num_samples = 0, 0, 0, 0
        n_batches = tf.data.experimental.cardinality(train_dataset).numpy()

        for iteration, (X_batch, y_batch) in enumerate(train_dataset, start=1):
            # Adjust learning rate and print
            current_lr = self.adjust_learning_rate(epoch, iteration, n_batches)
            print(f"Epoch {epoch} - Step {iteration}/{n_batches}, Learning Rate: {current_lr:.6f}")

            with tf.GradientTape() as tape:
                preds = self.model(X_batch, training=True)
                loss = self.loss_fn(y_batch, preds)
            grads = tape.gradient(loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            total_loss += loss
            total_top1 += tf.reduce_sum(self.compute_accuracy(y_batch, preds, top_k=1))
            total_top5 += tf.reduce_sum(self.compute_accuracy(y_batch, preds, top_k=5))
            num_samples += X_batch.shape[0]
        
        return total_loss / num_samples, total_top1 / num_samples, total_top5 / num_samples

    def validate(self, val_dataset):
        if self.stop_training:
            return 0, 0, 0  # Skip validation if early stopping is triggered
        
        total_loss, total_top1, total_top5, num_samples = 0, 0, 0, 0

        for X_batch, y_batch in val_dataset:
            preds = self.model(X_batch, training=False)
            loss = self.loss_fn(y_batch, preds)
            total_loss += loss
            total_top1 += tf.reduce_sum(self.compute_accuracy(y_batch, preds, top_k=1))
            total_top5 += tf.reduce_sum(self.compute_accuracy(y_batch, preds, top_k=5))
            num_samples += X_batch.shape[0]

        avg_top1 = total_top1 / num_samples
        print(f"Validation Top-1 Accuracy: {avg_top1:.4%}")

        # Early stopping check
        if avg_top1 >= self.early_stop_acc:
            print(f"Early Stopping Triggered: Top-1 Accuracy {avg_top1:.4%} >= {self.early_stop_acc:.4%}")
            self.stop_training = True

        return total_loss / num_samples, avg_top1, total_top5 / num_samples