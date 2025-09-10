"""
RetinoAI Pro - Main entry point for the application
"""

import argparse
import sys
from retinoai.app import create_app
from retinoai.train import train_oct_model, train_fundus_model

def main():
    parser = argparse.ArgumentParser(description="RetinoAI Pro - AI-Powered Retinal Disease Diagnosis")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Run web app command
    app_parser = subparsers.add_parser('app', help='Run the web application')
    app_parser.add_argument('--host', default='0.0.0.0', help='Host to run the app on')
    app_parser.add_argument('--port', type=int, default=5000, help='Port to run the app on')
    app_parser.add_argument('--debug', action='store_true', help='Run in debug mode')
    
    # Train models command
    train_parser = subparsers.add_parser('train', help='Train models')
    train_parser.add_argument('--model', choices=['oct', 'fundus', 'all'], default='all', 
                             help='Which model to train')
    train_parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    
    # Download data command
    download_parser = subparsers.add_parser('download', help='Download datasets')
    download_parser.add_argument('--dataset', choices=['oct', 'fundus', 'all'], default='all',
                                help='Which dataset to download')
    
    args = parser.parse_args()
    
    if args.command == 'app':
        app = create_app()
        app.run(host=args.host, port=args.port, debug=args.debug)
    
    elif args.command == 'train':
        if args.model in ['oct', 'all']:
            print("Training OCT model...")
            train_oct_model()
        
        if args.model in ['fundus', 'all']:
            print("Training Fundus model...")
            train_fundus_model()
    
    elif args.command == 'download':
        print("Dataset download functionality will be implemented soon.")
        print("Please download datasets manually from:")
        print("OCT: https://www.kaggle.com/datasets/obulisainaren/retinal-oct-c8")
        print("Fundus: https://www.kaggle.com/datasets/kssanjaynithish03/retinal-fundus-images")
    
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == "__main__":
    main()