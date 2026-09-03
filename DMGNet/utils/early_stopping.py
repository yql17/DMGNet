import torch

class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0.0,
                 model_path="../checkpoints/best_model.pt",
                 classifier_path="../checkpoints/best_classifier.pt"):
        """
        Early stops the training if validation score doesn't improve after a given patience.
        Also saves best weights of model and classifier.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta

        self.counter = 0
        self.epoch_count = 0
        self.best_epoch_num = 1
        self.best_score = None
        self.early_stop = False

        self.best_model_wts = None
        self.best_classifier_wts = None
        self.model_path = model_path
        self.classifier_path = classifier_path

    def __call__(self, val_score, model, classifier):
        self.epoch_count += 1

        if self.best_score is None:
            self.best_score = val_score
            self.best_epoch_num = self.epoch_count
            self.save_checkpoint(model, classifier)
        elif val_score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose and self.counter % 5 == 0:
                print(f"EarlyStopping counter: {self.counter} / {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_score
            self.best_epoch_num = self.epoch_count
            self.save_checkpoint(model, classifier)
            self.counter = 0
            if self.verbose:
                print(f"Validation score improved to {val_score:.4f}. Model saved.")

    def save_checkpoint(self, model, classifier):
        """
        Save the current best model and classifier weights to file and memory.
        """
        torch.save(model.state_dict(), self.model_path)
        torch.save(classifier.state_dict(), self.classifier_path)
        self.best_model_wts = model.state_dict()
        self.best_classifier_wts = classifier.state_dict()

    def restore_best_weights(self, model, classifier):
        """
        Restore the best weights to model and classifier from memory.
        """
        if self.best_model_wts and self.best_classifier_wts:
            model.load_state_dict(self.best_model_wts)
            classifier.load_state_dict(self.best_classifier_wts)
            if self.verbose:
                print(f"Restored model and classifier from epoch {self.best_epoch_num}.")
