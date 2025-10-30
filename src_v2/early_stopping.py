


class EarlyStopping:
    def __init__(self, patience=100):
        self.patience = patience
        self.best_loss = float('inf')
        self.count = 0
        self.early_stopping = False

    def __call__(self, loss):
        if loss < self.best_loss:
            self.best_loss = loss
            self.count = 0
        else:
            self.count += 1
            if self.count > self.patience:
                self.early_stopping = True
                print(f"Training Stopped Early")

        return self.early_stopping
    


if __name__ == '__main__':
    pass