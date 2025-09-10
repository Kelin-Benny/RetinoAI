# Add this to the end of your existing train.py file

def main():
    """Main function for training models"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train RetinoAI models')
    parser.add_argument('--model', type=str, choices=['oct', 'fundus', 'all'], 
                       default='all', help='Which model to train')
    parser.add_argument('--epochs', type=int, default=NUM_EPOCHS, 
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=LEARNING_RATE, 
                       help='Learning rate')
    
    args = parser.parse_args()
    
    if args.model in ['oct', 'all']:
        print("Training OCT model...")
        train_oct_model()
    
    if args.model in ['fundus', 'all']:
        print("Training Fundus model...")
        train_fundus_model()

if __name__ == '__main__':
    main()