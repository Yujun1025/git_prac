import torch
from torch import nn
from torch.nn import functional as F
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
from torch import optim
from torch.nn.modules.loss import CrossEntropyLoss
import nn_model
import loader
import numpy as np
import time
from torch.utils.data import Dataset, DataLoader, random_split
import os
import coral
import functions
import transformation
from torch.distributions import Categorical


def train(args):
    
    random_seed = args.random_seed
    np.random.seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    mode = args.training_mode
    test_set = args.test_data
    batch_size = args.batch_size
    nb_epochs = args.epochs
    trans = args.transformation
    train = args.train_data
    early_stop = functions.EarlyStopping(patience=200)
    if mode == 'baseline':
        
        acc = []
        if args.encoder == 'encoder_1':
            encoder = nn_model.encoder_1(1).to(device)
            cls_classifier = nn_model.class_classifer(4).to(device)
        
            optimizer = optim.SGD(
            list(encoder.parameters()) +
            list(cls_classifier.parameters()),
            lr=0.001, momentum = 0.9)

        elif args.encoder == 'encoder_2':
            encoder = nn_model.encoder_2(1).to(device)
            cls_classifier = nn_model.class_classifer(4).to(device)
            
            optimizer = optim.SGD(
            list(encoder.parameters()) +
            list(cls_classifier.parameters()),
            lr=0.001, momentum = 0.9)
        
        else:
            model = nn_model.res_sig(1).to(device)

            optimizer = optim.SGD(
            list(model.parameters()),
            lr=0.001, momentum = 0.9)
        path = '/home/workspace/checkpoint/' + args.name_exp + '_' + args.size_LD + '_' + str(args.random_seed)
        os.mkdir(path)
        early_stopping = functions.EarlyStopping(patience = 30, verbose = True, path = path)
       
        train_dataset = loader.CustomDataset(f'/home/workspace/dataset/train_data/{train}/set_{args.set_s}/label_{args.size_LD}.npy')
        # drop 은 data 개수가 batch 로 나누면 딱 1이어서 자꾸 에러나서
        valid_dataset = loader.CustomDataset(f'/home/workspace/dataset/valid_data/{train}/valid_set.npy')
        test_dataset = loader.CustomDataset(f'/home/workspace/dataset/test_data/{test_set}/test_set.npy')

        
        train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)
        
        nb_epochs = args.epochs
        
        for epoch in range(nb_epochs):
            for batch_idx, samples in enumerate(train_data_loader):
                x_train, y_train = samples
                x_train = x_train.to(device)
                y_train = y_train.to(device)
                
                if trans:

                    x_1 = transformation.augment(x_train, device)
                    x_2 = transformation.augment(x_train, device)
                    x_3 = transformation.augment(x_train, device)
                    x_4 = transformation.augment(x_train, device)
                    x_5 = transformation.augment(x_train, device)
                    x = torch.cat((x_train, x_1, x_2, x_3, x_4, x_5), 0)
                    x = x.to(device)
                    x = x.unsqueeze(1)
                    
                    if args.encoder == 'model':
                        prediction = model(x)
                    else:
                        features = encoder(x)
                        prediction = cls_classifier(features)
                        
                    y = torch.cat((y_train, y_train, y_train, y_train, y_train, y_train), axis = 0)
                    y = y.to(device)
                    cost = F.cross_entropy(prediction, y)
                    
                    optimizer.zero_grad()
                    cost.backward()
                    optimizer.step()
                
                else:
                    x_train = x_train.unsqueeze(1)
                    
                    if args.encoder == 'model':
                        prediction = model(x_train)
                    else:
                        features = encoder(x_train)
                        prediction = cls_classifier(features)
                    
                    
                    cost = F.cross_entropy(prediction, y_train)
                    optimizer.zero_grad()
                    cost.backward()
                    optimizer.step()
                    
            if (epoch+1) % 10 == 0:
                if args.encoder == 'model':
                    model.eval()
                else:
                    encoder.eval()
                    cls_classifier.eval()
                    
                
                test_loss = 0.0
                correct = 0

                class_correct = list(0. for i in range(4))
                class_total = list(0. for i in range(4))
                
                with torch.no_grad():
                    for images, labels in valid_data_loader:
                        images = images.to(device)
                        images = images.unsqueeze(1)
                        labels = labels.to(device)
                        if args.encoder == 'model':
                            outputs = model(images)
                        else:
                            features = encoder(images)
                            outputs = cls_classifier(features)
                        
                        cost = F.cross_entropy(outputs, labels)

                        test_loss += cost
  
                        predicted = torch.max(outputs, 1)[1]
                        labels = torch.max(labels, 1)[1]
                        correct += (labels == predicted).sum()

                        c = (predicted == labels).squeeze()
                        for i in range(len(labels)):
                            try:
                                class_correct[labels[i]] += c[i].item()
                            except:
                                pass
                            class_total[labels[i]] += 1
                len_total = class_total[0] + class_total[1] + class_total[2] + class_total[3]
                print('---------------- Validation Stage----------------------')  
                print(f'Epoch {epoch + 1} acc: {(correct / len_total)}')
                acc.append((correct / len_total))
                early_stopping(test_loss, encoder, cls_classifier)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break
    
        if args.encoder == 'encoder_1':
            encoder = nn_model.encoder_1(1).to(device)
            cls_classifier = nn_model.class_classifer(4).to(device)
        
            optimizer = optim.SGD(
            list(encoder.parameters()) +
            list(cls_classifier.parameters()),
            lr=0.001, momentum = 0.9)

        elif args.encoder == 'encoder_2':
            encoder = nn_model.encoder_2(1).to(device)
            cls_classifier = nn_model.class_classifer(4).to(device)
            
            optimizer = optim.SGD(
            list(encoder.parameters()) +
            list(cls_classifier.parameters()),
            lr=0.001, momentum = 0.9)
        
        checkpoint = torch.load(path + '/model.pt')
        
        encoder.load_state_dict(checkpoint['encoder'])
        cls_classifier.load_state_dict(checkpoint['classifier'])
        encoder.eval()
        cls_classifier.eval()
        test_loss = 0.0
        correct = 0
        with torch.no_grad():
            class_correct = list(0. for i in range(4))
            class_total = list(0. for i in range(4))
            for images, labels in test_dataloader:
                images = images.to(device)
                images = images.unsqueeze(1)
                labels = labels.to(device)
                
                features = encoder(images)
                predicted = cls_classifier(features)

                cost = F.cross_entropy(predicted, labels)
                predicted = torch.max(predicted, 1)[1]
                labels = torch.max(labels, 1)[1]
                correct += (labels == predicted).sum()
                test_loss += cost
                c = (predicted == labels).squeeze()
                for i in range(len(labels)):
                    class_correct[labels[i]] += c[i].item()
                    class_total[labels[i]] += 1
            total = class_total[0] + class_total[1] + class_total[2] + class_total[3]  
            print('--------------- Test Stage -----------------')
            print (f'cls_loss: {test_loss} acc: {(correct / total)}')
            print(f'class_0 {class_correct[0] / class_total[0]}')
            try:
                print(f'class_1 {class_correct[1] / class_total[1]}')
            except:
                print('No Ball bearing fault')
            print(f'class_2 {class_correct[2] / class_total[2]}')
            print(f'class_3 {class_correct[3] / class_total[3]}')
            acc.append((correct / total))

            for i in range(len(acc)):
                acc[i] = acc[i].cpu()
        
        os.mkdir(f'/home/workspace/Model/{args.name_exp}_{args.size_LD}_{random_seed}')
        torch.save(encoder.state_dict(), f'/home/workspace//Model/{args.name_exp}_{args.size_LD}_{random_seed}/encoder.pth')
        torch.save(cls_classifier.state_dict(), f'/home/workspace/Model/{args.name_exp}_{args.size_LD}_{random_seed}/cls_classifier.pth')
        np.save(f'/home/workspace/Model/{args.name_exp}_{args.size_LD}_{random_seed}/acc.npy', np.array(acc))

    elif mode == 'ssl':

        
         

        labeled_dataset = loader.CustomDataset(f'/home/workspace/dataset/train_data/{train}/set_{args.set_s}/label_{args.size_LD}.npy')
        unlabeled_dataset = loader.CustomDataset(f'/home/workspace/dataset/train_data/{train}/set_{args.set_s}/unlabeled_{args.size_UD}.npy')
        test_dataset = loader.CustomDataset(f'/home/workspace/dataset/test_data/{test_set}/test_set.npy')
        valid_dataset = loader.CustomDataset(f'/home/workspace/dataset/valid_data/{train}/valid_set.npy')
        path = '/home/workspace/checkpoint/' + args.name_exp + '_' + args.size_LD + '_' + str(args.random_seed)
        os.mkdir(path)
        early_stopping = functions.EarlyStopping(patience = 20, verbose = True, path = path)
        unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)
        labeled_loader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)

        encoder = nn_model.encoder_2(1).to(device)
        cls_classifier = nn_model.class_classifer(4).to(device)

        optimizer = optim.SGD(
        list(encoder.parameters()) +
        list(cls_classifier.parameters()),
        lr=0.001, momentum=0.9)
        
        acc = []    
        
        warmup = args.warmup
        if warmup:
            warm_epoch = nb_epochs // 4
        else:
            warm_epoch = 0
        for epoch in range(warm_epoch) :
            
            for batch_idx, samples in enumerate(labeled_loader):
                x_train, y_train = samples
                x_train = x_train.to(device)
                y_train = y_train.to(device)

                x_1 = transformation.augment(x_train, device)
                x_2 = transformation.augment(x_train, device)
                x_3 = transformation.augment(x_train, device)
                x_4 = transformation.augment(x_train, device)
                x = torch.cat((x_train, x_1, x_2, x_3, x_4), 0)
                x = x.to(device)
                x = x.unsqueeze(1)
                features = encoder(x)
                prediction = cls_classifier(features)

                y = torch.cat((y_train, y_train, y_train, y_train, y_train), axis = 0)

                cost = F.cross_entropy(prediction, y)
                
                optimizer.zero_grad()
                cost.backward()
                optimizer.step()
                
            
            if (epoch+1) % 10 == 0:
                    encoder.eval()
                    cls_classifier.eval()
                    with torch.no_grad():
                        
                        test_loss = 0.0
                        correct = 0

                        class_correct = list(0. for i in range(4))
                        class_total = list(0. for i in range(4))
                        for images, labels in valid_loader:
                            images = images.to(device)
                            images = images.unsqueeze(1)
                            labels = labels.to(device)
                            
                            features = encoder(images)
                            outputs = cls_classifier(features)

                            cost = F.cross_entropy(outputs, labels)
                            predicted = torch.max(outputs, 1)[1]
                            labels = torch.max(labels, 1)[1]
                            correct += (labels == predicted).sum()
                            test_loss += cost
                            c = (predicted == labels).squeeze()
                            for i in range(len(labels)):
                                class_correct[labels[i]] += c[i].item()
                                class_total[labels[i]] += 1
                    total = class_total[0] + class_total[1] + class_total[2] + class_total[3]  
                    print('--------------- validation stage -----------------')
                    print (f'epoch{epoch + 1}, cls_loss: {test_loss}, acc: {(correct / total)}')
                    print(f'class_0 {class_correct[0] / class_total[0]}')
                    try:
                        print(f'class_1 {class_correct[1] / class_total[1]}')
                    except:
                        print('No Ball bearing fault')
                    print(f'class_2 {class_correct[2] / class_total[2]}')
                    print(f'class_3 {class_correct[3] / class_total[3]}')
                    acc.append((correct / total))
        

        optimizer = optim.Adam(
        list(encoder.parameters()) +
        list(cls_classifier.parameters()),
        lr=0.00001)  
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0005)
        for epoch in range(warm_epoch, nb_epochs):
            lambda_1 =  args.lambda_1 * ((epoch / nb_epochs) **2 )
            if epoch < (nb_epochs / 2):
                tau = args.tau - 1
            else:
                tau = args.tau + 1
            for batch_idx, (labeled, unlabeled) in enumerate(zip(labeled_loader, unlabeled_loader)):
                               
                x_train, y_train = labeled
                x_train = x_train.to(device)
                y_train = y_train.to(device)

                x_1 = transformation.augment(x_train, device)
                x_2 = transformation.augment(x_train, device)
                x_3 = transformation.augment(x_train, device)
                x_4 = transformation.augment(x_train, device)
                x = torch.cat((x_train, x_1, x_2, x_3, x_4), 0)
                x = x.to(device)
                x = x.unsqueeze(1)
                
                features = encoder(x)
                prediction = cls_classifier(features)
                
                y = torch.cat((y_train, y_train, y_train, y_train, y_train), axis = 0)
                labeled_cost = F.cross_entropy(prediction, y).to(device)
                
                x_un, _ = unlabeled
                x_un = x_un.to(device)

                x_1 = transformation.augment(x_un, device)
                x_1 = x_1.unsqueeze(1)
                
                x_2 = transformation.augment(x_un, device)
                x_2 = x_2.unsqueeze(1)
                
                x_3 = transformation.augment(x_un, device)
                x_3 = x_3.unsqueeze(1)


                features_1 = encoder(x_1)
                prediction_1 = cls_classifier(features_1)

                features_2 = encoder(x_2)
                prediction_2 = cls_classifier(features_2)

                features_3 = encoder(x_3)
                prediction_3 = cls_classifier(features_3)
                
                pseudo = (prediction_1 + prediction_2 + prediction_3) / 3
                                
                with torch.no_grad():
                    pseudo_label = functions.sharpen(pseudo, tau).to(device)

                un_cost_1 = F.mse_loss(prediction_1, pseudo_label).to(device)
                un_cost_2 = F.mse_loss(prediction_2, pseudo_label).to(device)
                un_cost_3 = F.mse_loss(prediction_3, pseudo_label).to(device)

                unlabeled_cost = un_cost_1 + un_cost_2 + un_cost_3
                
                total_cost = labeled_cost + lambda_1 * unlabeled_cost
                optimizer.zero_grad()
                total_cost.backward()
                optimizer.step()
            scheduler.step()
            if (epoch+1) % 10 == 0:
                encoder.eval()
                cls_classifier.eval()
                with torch.no_grad():
                    
                    test_loss = 0.0
                    correct = 0

                    class_correct = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    for images, labels in valid_loader:
                        images = images.to(device)
                        images = images.unsqueeze(1)
                        labels = labels.to(device)
                        
                        features = encoder(images)
                        outputs = cls_classifier(features)

                        cost = F.cross_entropy(outputs, labels)
                        predicted = torch.max(outputs, 1)[1]
                        labels = torch.max(labels, 1)[1]
                        correct += (labels == predicted).sum()
                        test_loss += cost
                        c = (predicted == labels).squeeze()
                        for i in range(len(labels)):
                            class_correct[labels[i]] += c[i].item()
                            class_total[labels[i]] += 1
                total = class_total[0] + class_total[1] + class_total[2] + class_total[3]  
                print('--------------- validation stage -----------------')
                print (f'epoch{epoch + 1}, cls_loss: {test_loss}, acc: {(correct / total)}')
                print(f'class_0 {class_correct[0] / class_total[0]}')
                try:
                    print(f'class_1 {class_correct[1] / class_total[1]}')
                except:
                    print('No Ball bearing fault')
                print(f'class_2 {class_correct[2] / class_total[2]}')
                print(f'class_3 {class_correct[3] / class_total[3]}')
                acc.append((correct / total))
        
                early_stopping(test_loss, encoder, cls_classifier)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break
    
        if args.encoder == 'encoder_1':
            encoder = nn_model.encoder_1(1).to(device)
            cls_classifier = nn_model.class_classifer(4).to(device)
        
            optimizer = optim.SGD(
            list(encoder.parameters()) +
            list(cls_classifier.parameters()),
            lr=0.001, momentum = 0.9)

        elif args.encoder == 'encoder_2':
            encoder = nn_model.encoder_2(1).to(device)
            cls_classifier = nn_model.class_classifer(4).to(device)
            
            optimizer = optim.SGD(
            list(encoder.parameters()) +
            list(cls_classifier.parameters()),
            lr=0.001, momentum = 0.9)
        
        checkpoint = torch.load(path + '/model.pt')
        
        encoder.load_state_dict(checkpoint['encoder'])
        cls_classifier.load_state_dict(checkpoint['classifier'])
        encoder.eval()
        cls_classifier.eval()
        
        with torch.no_grad():
            
            test_loss = 0.0
            correct = 0

            class_correct = list(0. for i in range(4))
            class_total = list(0. for i in range(4))
            for images, labels in test_dataloader:
                images = images.to(device)
                images = images.unsqueeze(1)
                labels = labels.to(device)
                
                features = encoder(images)
                predicted = cls_classifier(features)

                cost = F.cross_entropy(predicted, labels)
                predicted = torch.max(predicted, 1)[1]
                labels = torch.max(labels, 1)[1]
                correct += (labels == predicted).sum()
                test_loss += cost
                c = (predicted == labels).squeeze()
                for i in range(len(labels)):
                    class_correct[labels[i]] += c[i].item()
                    class_total[labels[i]] += 1
        total = class_total[0] + class_total[1] + class_total[2] + class_total[3]  
        print('--------------- Test Stage -----------------')
        print (f'cls_loss: {test_loss} acc: {(correct / total)}')
        print(f'class_0 {class_correct[0] / class_total[0]}')
        try:
            print(f'class_1 {class_correct[1] / class_total[1]}')
        except:
            print('No Ball bearing fault')
        print(f'class_2 {class_correct[2] / class_total[2]}')
        print(f'class_3 {class_correct[3] / class_total[3]}')
        acc.append((correct / total))
       
        for i in range(len(acc)):
            acc[i] = acc[i].cpu()
        
        os.mkdir(f'/home/workspace/Model/{args.name_exp}_{args.size_LD}_{random_seed}')
        torch.save(encoder.state_dict(), f'/home/workspace//Model/{args.name_exp}_{args.size_LD}_{random_seed}/encoder.pth')
        torch.save(cls_classifier.state_dict(), f'/home/workspace/Model/{args.name_exp}_{args.size_LD}_{random_seed}/cls_classifier.pth')
        np.save(f'/home/workspace/Model/{args.name_exp}_{args.size_LD}_{random_seed}/acc.npy', np.array(acc))
    
    
    elif mode == 'multi_domain':
        multi_domain = args.multi_domain

        target_dataset = loader.CustomDataset(f'/home/workspace/dataset/train_data/{train}/set_{args.set_s}/label_{args.size_LD}.npy')
        unlabeled_target_dataset = loader.CustomDataset(f'/home/workspace/dataset/train_data/{train}/set_{args.set_s}/unlabeled_{args.size_UD}.npy')
        multi_domain_data = loader.CustomDataset(f'/home/workspace/dataset/train_data/{multi_domain}/set_{args.set_m}/unlabeled_{args.size_MDU}.npy')
        labeled_multi_domain_data = loader.CustomDataset(f'/home/workspace/dataset/train_data/{multi_domain}/set_{args.set_m}/label_{args.size_MDL}.npy')
        multi_valid_dataset = loader.CustomDataset(f'/home/workspace/dataset/valid_data/{multi_domain}/valid_set.npy')
        valid_dataset = loader.CustomDataset(f'/home/workspace/dataset/valid_data/{train}/valid_set.npy')
        test_dataset = loader.CustomDataset(f'/home/workspace/dataset/test_data/{multi_domain}/test_set.npy')
        
        
        train_target_dataloader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)
        unlabeled_loader = DataLoader(unlabeled_target_dataset, batch_size=batch_size, shuffle=True)
        
        multi_domain_data_loader = DataLoader(multi_domain_data, batch_size= batch_size, shuffle= True)
        multi_domain_labeled_data_loader = DataLoader(labeled_multi_domain_data, batch_size= batch_size, shuffle= True)
        
        multi_valid_data_loader = DataLoader(multi_valid_dataset, batch_size=batch_size, shuffle=True)
        valid_data_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        
        test_dataloader = DataLoader(test_dataset, batch_size=8, shuffle=False)      

        encoder = nn_model.encoder_2(1).to(device)
        cls_classifier = nn_model.class_classifer(4).to(device)
        domain_classifier = nn_model.domain_classifer().to(device)
        
        path = '/home/workspace/checkpoint/' + args.name_exp + '_' + args.size_LD + '_' + str(args.random_seed)
        os.mkdir(path)
        early_stopping = functions.EarlyStopping(patience = 20, verbose = True, delta = 0.0005, path = path)

        os.mkdir(f'/home/workspace/Model/{args.name_exp}_{args.size_LD}_{random_seed}')

        optimizer = optim.Adam(
        list(encoder.parameters()) +
        list(cls_classifier.parameters()) +
        list(domain_classifier.parameters()),
        lr=0.0005)
       

        acc = []

        warmup = args.warmup
        if warmup:
            warm_epoch = nb_epochs // 4
        else:
            warm_epoch = 0
        for epoch in range(warm_epoch) :
            p = float(epoch / nb_epochs) 
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            p = float(epoch / nb_epochs) 
            alpha = p

            lambda_1 =  args.lambda_1 * ((epoch / nb_epochs) ** 2)
            lambda_2 =  args.lambda_1 * (epoch / nb_epochs)

            for batch_idx, (source_samples, source_un_samples, target_labeled, target_samples) in enumerate(zip(train_target_dataloader,unlabeled_loader,multi_domain_labeled_data_loader, multi_domain_data_loader)):
                x_train, y_train = source_samples
                x_train = x_train.to(device)
                y_train = y_train.to(device)

                x_1 = transformation.augment(x_train, device)
                x_2 = transformation.augment(x_train, device)
                x_3 = transformation.augment(x_train, device)
                x_4 = transformation.augment(x_train, device)

                x = torch.cat((x_train, x_1, x_2, x_3, x_4), 0)
                x = x.to(device)
                x = x.unsqueeze(1)

                features = encoder(x)
                prediction = cls_classifier(features)
                
                y = torch.cat((y_train, y_train, y_train, y_train, y_train), axis = 0)
                
                labeled_cost = F.cross_entropy(prediction, y).to(device)

                unlabeled_target, _ = source_un_samples
                unlabeled_target = unlabeled_target.to(device)
                u_2 = transformation.augment(unlabeled_target, device)
                u_3 = transformation.augment(unlabeled_target, device)
                
                unlabeled_target =unlabeled_target.unsqueeze(1)
                u_2 = u_2.unsqueeze(1)
                u_3 = u_3.unsqueeze(1)

                u_features_1 = encoder(unlabeled_target)
                u_pred_1 = cls_classifier(u_features_1)
                
                u_features_2 = encoder(u_2)
                u_pred_2 = cls_classifier(u_features_2)

                u_features_3 = encoder(u_3)
                u_pred_3 = cls_classifier(u_features_3)

                u_pseudo = (u_pred_1 + u_pred_2 + u_pred_3) / 3
                with torch.no_grad():
                    u_pseudo_label = functions.sharpen(u_pseudo, tau).to(device)

                u_cost_1 = F.mse_loss(u_pred_1, u_pseudo_label).to(device)
                u_cost_2 = F.mse_loss(u_pred_2, u_pseudo_label).to(device)
                u_cost_3 = F.mse_loss(u_pred_3, u_pseudo_label).to(device)
                
                ml_x, ml_y = target_labeled
                ml_x = ml_x.to(device)
                
                ml_2 = transformation.augment(ml_x, device)
                ml_3 = transformation.augment(ml_x, device)
                ml_4 = transformation.augment(ml_x, device)
                ml_5 = transformation.augment(ml_x, device)

                m = torch.cat((ml_x, ml_2, ml_3, ml_4, ml_5), axis = 0)
                m = m.unsqueeze(1)
                ml_y = ml_y.to(device)
                mly = torch.cat((ml_y, ml_y, ml_y, ml_y, ml_y), axis = 0)

                d_features = encoder(m)
                d_prediction = cls_classifier(d_features)

                dl_cost = F.cross_entropy(d_prediction, mly)

                m_domain_img, _ = target_samples
                m_domain_img = m_domain_img.to(device)
                m_2 = transformation.augment(m_domain_img, device)
                m_3 = transformation.augment(m_domain_img, device)

                m_domain_img = m_domain_img.unsqueeze(1)
                m_2 = m_2.unsqueeze(1)
                m_3 = m_3.unsqueeze(1)

                m_features_1 = encoder(m_domain_img)

                m_features_2 = encoder(m_2)

                m_features_3 = encoder(m_3)

                m_features = torch.cat((m_features_1, m_features_2, m_features_3), axis = 0)
                
                tar_d = torch.cat((u_features_1, features), axis = 0).mean(dim = 0)
                sub_d = torch.cat((d_features, m_features), axis = 0).mean(dim = 0)

                mmd_loss = F.mse_loss(tar_d, sub_d)
                # for domain alignment
        
                
                u_cost = u_cost_1 + u_cost_2 + u_cost_3

                u_pseudo_max = torch.max(u_pseudo_label, 1)[1]
                y_max = torch.max(y, 1)[1]
                
            
                total_cost = labeled_cost + dl_cost + lambda_2 * u_cost
                
                
                
                p_target_features = torch.cat((u_features_1, u_features_2), axis = 0)
                p_target_label = torch.cat((u_pseudo_max, u_pseudo_max), axis = 0)
                

                optimizer.zero_grad()
                total_cost.backward(retain_graph = True)
                optimizer.step()
                
                # fix classifier and update encoder
                for param in cls_classifier.parameters():
                    param.requires_grad = False
                torch.autograd.set_detect_anomaly(True)
                
                source_features = torch.cat((features, u_features_1, u_features_2), axis = 0)
                target_features = torch.cat((d_features, m_features_1, m_features_2), axis = 0)
                
                # zero_s = torch.zeros(source_features.size(0), 1).to(device)
                # one_s = torch.ones(source_features.size(0), 1).to(device)
                # source_label = torch.cat((zero_s, one_s), axis = 1)

                # zero_t = torch.zeros(target_features.size(0), 1).to(device)
                # one_t = torch.ones(target_features.size(0), 1).to(device)
                # target_label = torch.cat((one_t, zero_t), axis = 1)

                # source_pred = domain_classifier(source_features, alpha = alpha)
                # target_pred = domain_classifier(target_features, alpha = alpha)

                # source_loss = F.cross_entropy(source_pred, source_label).to(device)
                # target_loss = F.cross_entropy(target_pred, target_label).to(device)

                # domain_loss = source_loss + target_loss
                
                p_target_class_1 = p_target_features[torch.where(p_target_label == 0)].mean(axis = 0) 
                p_target_class_2 = p_target_features[torch.where(p_target_label == 1)].mean(axis = 0) 
                p_target_class_3 = p_target_features[torch.where(p_target_label == 2)].mean(axis = 0) 
                p_target_class_4 = p_target_features[torch.where(p_target_label == 3)].mean(axis = 0) 
                
                r_target_class_1 = features[torch.where(y_max == 0)].mean(axis = 0)
                r_target_class_2 = features[torch.where(y_max == 1)].mean(axis = 0)
                r_target_class_3 = features[torch.where(y_max == 2)].mean(axis = 0)
                r_target_class_4 = features[torch.where(y_max == 3)].mean(axis = 0)

                
                m_label_max = torch.max(mly, 1)[1]

                
                p_sub_features = torch.cat((m_features_1, m_features_2), axis = 0)

                r_sub_class_1 = d_features[torch.where(m_label_max == 0)].mean(axis = 0)
                r_sub_class_2 = d_features[torch.where(m_label_max == 1)].mean(axis = 0)
                r_sub_class_3 = d_features[torch.where(m_label_max == 2)].mean(axis = 0)
                r_sub_class_4 = d_features[torch.where(m_label_max == 3)].mean(axis = 0)

                constrastive_loss = functions.ContrastiveLoss()
                con_val = constrastive_loss(r_target_class_1, r_sub_class_1, 0) + constrastive_loss(r_target_class_2, r_sub_class_2, 0) + \
                            constrastive_loss(r_target_class_3, r_sub_class_3, 0) + constrastive_loss(r_target_class_4, r_sub_class_4, 0)
                
                real_con = constrastive_loss(r_target_class_1, r_sub_class_1, 0) + constrastive_loss(r_target_class_1, r_sub_class_2, 1) + \
                            constrastive_loss(r_target_class_1, r_sub_class_3, 1) + constrastive_loss(r_target_class_1, r_sub_class_4, 1) + \
                            constrastive_loss(r_target_class_2, r_sub_class_2, 0) + constrastive_loss(r_target_class_2, r_sub_class_3, 1) + \
                            constrastive_loss(r_target_class_2, r_sub_class_4, 1) + constrastive_loss(r_target_class_3, r_sub_class_3, 0) + \
                            constrastive_loss(r_target_class_3, r_sub_class_4, 1) + constrastive_loss(r_target_class_4, r_sub_class_4, 0)
                        
                # adapt_con = constrastive_loss(r_target_class_1, p_sub_class_1, 0) + constrastive_loss(r_target_class_2, p_sub_class_2, 0) + \
                #             constrastive_loss(r_target_class_3, p_sub_class_3, 0) + constrastive_loss(r_target_class_4, p_sub_class_4, 0)
                
                adapt_con = constrastive_loss(p_target_class_1, r_sub_class_1, 0) + constrastive_loss(p_target_class_2, r_sub_class_2, 0) + \
                            constrastive_loss(p_target_class_3, r_sub_class_3, 0) + constrastive_loss(p_target_class_4, r_sub_class_4, 0)

                
                intra_con = constrastive_loss(r_target_class_1, r_target_class_2, 1) + constrastive_loss(r_target_class_1, r_target_class_3, 1) + \
                            constrastive_loss(r_target_class_1, r_target_class_4, 1) + constrastive_loss(r_target_class_2, r_target_class_3, 1) + \
                            constrastive_loss(r_target_class_2, r_target_class_4, 1) + constrastive_loss(r_target_class_3, r_target_class_4, 1)    

                con_loss = mmd_loss + real_con + 0.8 * (epoch / nb_epochs) * adapt_con + 0.8 * intra_con

                optimizer.zero_grad()
                con_loss.requires_grad_(True)
                con_loss.backward()
                optimizer.step()
                
                for param in cls_classifier.parameters():
                    param.requires_grad = True
            

            if (epoch+1) % 10 == 0:
                encoder.eval()
                cls_classifier.eval()
                with torch.no_grad():
                    
                    test_loss = 0.0
                    correct = 0
                    m_test_loss = 0.0
                    m_correct = 0

                    class_correct = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    for images, labels in valid_data_loader:
                        images = images.to(device)
                        images = images.unsqueeze(1)
                        labels = labels.to(device)
                        
                        features = encoder(images)
                        outputs = cls_classifier(features)

                        cost = F.cross_entropy(outputs, labels)
                        predicted = torch.max(outputs, 1)[1]
                        labels = torch.max(labels, 1)[1]
                        correct += (labels == predicted).sum()
                        test_loss += cost
                        c = (predicted == labels).squeeze()
                        for i in range(len(labels)):
                            class_correct[labels[i]] += c[i].item()
                            class_total[labels[i]] += 1
                    
                    m_class_correct = list(0. for i in range(4))
                    m_class_total = list(0. for i in range(4))
                    
                    for images, labels in multi_valid_data_loader:
                        images = images.to(device)
                        images = images.unsqueeze(1)
                        labels = labels.to(device)
                        
                        features = encoder(images)
                        outputs = cls_classifier(features)

                        cost = F.cross_entropy(outputs, labels)
                        predicted = torch.max(outputs, 1)[1]
                        labels = torch.max(labels, 1)[1]
                        m_correct += (labels == predicted).sum()
                        m_test_loss += cost
                        c = (predicted == labels).squeeze()
                        for i in range(len(labels)):
                            m_class_correct[labels[i]] += c[i].item()
                            m_class_total[labels[i]] += 1
                print('--------------- validation stage -----------------')
                total = class_total[0] + class_total[1] + class_total[2] + class_total[3]  
                print (f'epoch{epoch + 1}, cls_loss: {test_loss}, acc: {(correct / total)}')
                print(f'class_0 {class_correct[0] / class_total[0]}')
                try:
                    print(f'class_1 {class_correct[1] / class_total[1]}')
                except:
                    print('No Ball bearing fault')
                print(f'class_2 {class_correct[2] / class_total[2]}')
                print(f'class_3 {class_correct[3] / class_total[3]}')
                print('mmd_loss',mmd_loss)
                
                m_total = m_class_total[0] + m_class_total[1] + m_class_total[2] + m_class_total[3]
                print('--------------- multi domain validation stage -----------------')  
                print (f'epoch{epoch + 1}, cls_loss: {m_test_loss}, acc: {(m_correct / m_total)}')
                print(f'class_0 {m_class_correct[0] / m_class_total[0]}')
                try:
                    print(f'class_1 {m_class_correct[1] / m_class_total[1]}')
                except:
                    print('No ball bearing fault')
                print(f'class_2 {m_class_correct[2] / m_class_total[2]}')
                print(f'class_3 {m_class_correct[3] / m_class_total[3]}')

                acc.append((correct / total))

        # torch.save(encoder.state_dict(), f'/home/data/CWRU/Model/{args.name_exp}_{args.size_LD}/warmup_encoder.pth')
        # torch.save(cls_classifier.state_dict(), f'/home/data/CWRU/Model/{args.name_exp}_{args.size_LD}/warmpup_classifier.pth')
         
        
        optimizer = optim.Adam(
        list(encoder.parameters()) +
        list(cls_classifier.parameters()) +
        list(domain_classifier.parameters()),
        lr=0.0001)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.0005)
        for epoch in range(warm_epoch, nb_epochs):
            p = float(epoch / nb_epochs) 
            alpha = p

            lambda_1 =  args.lambda_1 * ((epoch / nb_epochs) ** 2)
            lambda_2 =  args.lambda_1 * (epoch / nb_epochs)

            if epoch < (nb_epochs / 2):
                tau = args.tau - 1
            else:
                tau = args.tau + 1

            for batch_idx, (source_samples, source_un_samples, target_labeled, target_samples) in enumerate(zip(train_target_dataloader,unlabeled_loader,multi_domain_labeled_data_loader, multi_domain_data_loader)):
                for param in cls_classifier.parameters():
                    param.requires_grad = False
                x_train, y_train = source_samples
                x_train = x_train.to(device)
                y_train = y_train.to(device)

                x_1 = transformation.augment(x_train, device)
                x_2 = transformation.augment(x_train, device)
                x_3 = transformation.augment(x_train, device)
                x_4 = transformation.augment(x_train, device)

                x = torch.cat((x_train, x_1, x_2, x_3, x_4), 0)
                x = x.to(device)
                x = x.unsqueeze(1)

                features = encoder(x)
                prediction = cls_classifier(features)
                
                y = torch.cat((y_train, y_train, y_train, y_train, y_train), axis = 0)
                
                labeled_cost = F.cross_entropy(prediction, y).to(device)

                unlabeled_target, _ = source_un_samples
                unlabeled_target = unlabeled_target.to(device)
                u_2 = transformation.augment(unlabeled_target, device)
                u_3 = transformation.augment(unlabeled_target, device)
                
                unlabeled_target =unlabeled_target.unsqueeze(1)
                u_2 = u_2.unsqueeze(1)
                u_3 = u_3.unsqueeze(1)

                u_features_1 = encoder(unlabeled_target)
                s_prediction_1 = cls_classifier(u_features_1)
                
                u_features_2 = encoder(u_2)
                s_prediction_2 = cls_classifier(u_features_2)

                u_features_3 = encoder(u_3)
                s_prediction_3 = cls_classifier(u_features_3)

                s_pseudo = (s_prediction_1 + s_prediction_2 + s_prediction_3) / 3
                with torch.no_grad():
                    s_pseudo_label = functions.sharpen(s_pseudo, tau).to(device)

                u_cost_1 = F.mse_loss(s_prediction_1, s_pseudo_label).to(device)
                u_cost_2 = F.mse_loss(s_prediction_2, s_pseudo_label).to(device)
                u_cost_3 = F.mse_loss(s_prediction_3, s_pseudo_label).to(device)
                
                ml_x, ml_y = target_labeled
                ml_x = ml_x.to(device)
                
                ml_2 = transformation.augment(ml_x, device)
                ml_3 = transformation.augment(ml_x, device)
                ml_4 = transformation.augment(ml_x, device)
                ml_5 = transformation.augment(ml_x, device)

                m = torch.cat((ml_x, ml_2, ml_3, ml_4, ml_5), axis = 0)
                m = m.unsqueeze(1)
                ml_y = ml_y.to(device)
                mly = torch.cat((ml_y, ml_y, ml_y, ml_y, ml_y), axis = 0)

                d_features = encoder(m)
                d_prediction = cls_classifier(d_features)

                dl_cost = F.cross_entropy(d_prediction, mly)

                m_domain_img, _ = target_samples
                m_domain_img = m_domain_img.to(device)
                m_2 = transformation.augment(m_domain_img, device)
                m_3 = transformation.augment(m_domain_img, device)

                m_domain_img = m_domain_img.unsqueeze(1)
                m_2 = m_2.unsqueeze(1)
                m_3 = m_3.unsqueeze(1)

                m_features_1 = encoder(m_domain_img)
                m_predictions_1 = cls_classifier(m_features_1)

                m_features_2 = encoder(m_2)
                m_predictions_2 = cls_classifier(m_features_2)

                m_features_3 = encoder(m_3)
                m_predictions_3 = cls_classifier(m_features_3)

                m_features = torch.cat((m_features_1, m_features_2, m_features_3), axis = 0)
                 
                tar_d = torch.cat((u_features_1, features), axis = 0).mean(dim = 0)
                sub_d = torch.cat((d_features, m_features), axis = 0).mean(dim = 0)

                mmd_loss = F.mse_loss(tar_d, sub_d)
                # for domain alignment

                m_pseudo = (m_predictions_1 + m_predictions_2 + m_predictions_3) / 3
                
                with torch.no_grad():
                    m_pseudo_label = functions.sharpen(m_pseudo, tau).to(device)

                m_cost_1 = F.mse_loss(m_predictions_1, m_pseudo_label).to(device)
                m_cost_2 = F.mse_loss(m_predictions_2, m_pseudo_label).to(device)   
                m_cost_3 = F.mse_loss(m_predictions_3, m_pseudo_label).to(device)                
                
                u_cost = u_cost_1 + u_cost_2 + u_cost_3
                m_cost = m_cost_1 + m_cost_2 + m_cost_3

                # u_pseudo_max = torch.max(u_pseudo_label, 1)[1]
                y_max = torch.max(y, 1)[1]
                
                # p_target_features = torch.cat((u_features_1, u_features_2), axis = 0)
                # p_target_label = torch.cat((u_pseudo_max, u_pseudo_max), axis = 0)
                # p_target_class_1 = p_target_features[torch.where(p_target_label == 0)].mean(axis = 0) 
                # p_target_class_2 = p_target_features[torch.where(p_target_label == 1)].mean(axis = 0) 
                # p_target_class_3 = p_target_features[torch.where(p_target_label == 2)].mean(axis = 0) 
                # p_target_class_4 = p_target_features[torch.where(p_target_label == 3)].mean(axis = 0) 
                
                r_target_class_1 = features[torch.where(y_max == 0)].mean(axis = 0)
                r_target_class_2 = features[torch.where(y_max == 1)].mean(axis = 0)
                r_target_class_3 = features[torch.where(y_max == 2)].mean(axis = 0)
                r_target_class_4 = features[torch.where(y_max == 3)].mean(axis = 0)

                
                m_label_max = torch.max(mly, 1)[1]
                # m_pseudo_max = torch.max(m_pseudo_label, 1)[1]
                
                # p_sub_features = torch.cat((m_features_1, m_features_2), axis = 0)
                # p_sub_label = torch.cat((m_pseudo_max, m_pseudo_max), axis = 0)

                # p_sub_class_1 = p_sub_features[torch.where(p_sub_label == 0)].mean(axis = 0)
                # p_sub_class_2 = p_sub_features[torch.where(p_sub_label == 1)].mean(axis = 0)  
                # p_sub_class_3 = p_sub_features[torch.where(p_sub_label == 2)].mean(axis = 0) 
                # p_sub_class_4 = p_sub_features[torch.where(p_sub_label == 3)].mean(axis = 0) 

                r_sub_class_1 = d_features[torch.where(m_label_max == 0)].mean(axis = 0)
                r_sub_class_2 = d_features[torch.where(m_label_max == 1)].mean(axis = 0)
                r_sub_class_3 = d_features[torch.where(m_label_max == 2)].mean(axis = 0)
                r_sub_class_4 = d_features[torch.where(m_label_max == 3)].mean(axis = 0)

                constrastive_loss = functions.ContrastiveLoss()
                # con_val = constrastive_loss(r_target_class_1, r_sub_class_1, 0) + constrastive_loss(r_target_class_2, r_sub_class_2, 0) + \
                #             constrastive_loss(r_target_class_3, r_sub_class_3, 0) + constrastive_loss(r_target_class_4, r_sub_class_4, 0)
                
                real_con = constrastive_loss(r_target_class_1, r_sub_class_1, 0) + constrastive_loss(r_target_class_1, r_sub_class_2, 1) + \
                            constrastive_loss(r_target_class_1, r_sub_class_3, 1) + constrastive_loss(r_target_class_1, r_sub_class_4, 1) + \
                            constrastive_loss(r_target_class_2, r_sub_class_2, 0) + constrastive_loss(r_target_class_2, r_sub_class_3, 1) + \
                            constrastive_loss(r_target_class_2, r_sub_class_4, 1) + constrastive_loss(r_target_class_3, r_sub_class_3, 0) + \
                            constrastive_loss(r_target_class_3, r_sub_class_4, 1) + constrastive_loss(r_target_class_4, r_sub_class_4, 0)
                        
                # adapt_con = constrastive_loss(r_target_class_1, p_sub_class_1, 0) + constrastive_loss(r_target_class_2, p_sub_class_2, 0) + \
                #             constrastive_loss(r_target_class_3, p_sub_class_3, 0) + constrastive_loss(r_target_class_4, p_sub_class_4, 0)
                
                # adapt_con = constrastive_loss(p_target_class_1, r_sub_class_1, 0) + constrastive_loss(p_target_class_2, r_sub_class_2, 0) + \
                #             constrastive_loss(p_target_class_3, r_sub_class_3, 0) + constrastive_loss(p_target_class_4, r_sub_class_4, 0)

                
                # intra_con = constrastive_loss(r_target_class_1, r_target_class_2, 1) + constrastive_loss(r_target_class_1, r_target_class_3, 1) + \
                #             constrastive_loss(r_target_class_1, r_target_class_4, 1) + constrastive_loss(r_target_class_2, r_target_class_3, 1) + \
                #             constrastive_loss(r_target_class_2, r_target_class_4, 1) + constrastive_loss(r_target_class_3, r_target_class_4, 1)    

                total_cost = labeled_cost + dl_cost + lambda_2 * u_cost + lambda_1 * m_cost + mmd_loss + 5 * real_con
                
                

                 

                optimizer.zero_grad()
                total_cost.backward(retain_graph = True)
                optimizer.step()
                for param in cls_classifier.parameters():
                    param.requires_grad = True
                # fix classifier and update encoder
                for param in encoder.parameters():
                    param.requires_grad = False
                torch.autograd.set_detect_anomaly(True)
                
                # source_features = torch.cat((features, u_features_1, u_features_2), axis = 0)
                # target_features = torch.cat((d_features, m_features_1, m_features_2), axis = 0)
                
                # zero_s = torch.zeros(source_features.size(0), 1).to(device)
                # one_s = torch.ones(source_features.size(0), 1).to(device)
                # source_label = torch.cat((zero_s, one_s), axis = 1)

                # zero_t = torch.zeros(target_features.size(0), 1).to(device)
                # one_t = torch.ones(target_features.size(0), 1).to(device)
                # target_label = torch.cat((one_t, zero_t), axis = 1)

                # source_pred = domain_classifier(source_features, alpha = alpha)
                # target_pred = domain_classifier(target_features, alpha = alpha)

                # source_loss = F.cross_entropy(source_pred, source_label).to(device)
                # target_loss = F.cross_entropy(target_pred, target_label).to(device)

                # domain_loss = source_loss + target_loss
                
                # p_target_features = torch.cat((u_features_1, u_features_2), axis = 0)
                # p_target_label = torch.cat((u_pseudo_max, u_pseudo_max), axis = 0)
                
                # p_target_class_1 = p_target_features[torch.where(p_target_label == 0)].mean(axis = 0) 
                # p_target_class_2 = p_target_features[torch.where(p_target_label == 1)].mean(axis = 0) 
                # p_target_class_3 = p_target_features[torch.where(p_target_label == 2)].mean(axis = 0) 
                # p_target_class_4 = p_target_features[torch.where(p_target_label == 3)].mean(axis = 0) 
                
                # r_target_class_1 = features[torch.where(y_max == 0)].mean(axis = 0)
                # r_target_class_2 = features[torch.where(y_max == 1)].mean(axis = 0)
                # r_target_class_3 = features[torch.where(y_max == 2)].mean(axis = 0)
                # r_target_class_4 = features[torch.where(y_max == 3)].mean(axis = 0)

                
                # m_label_max = torch.max(mly, 1)[1]
                # m_pseudo_max = torch.max(m_pseudo_label, 1)[1]
                
                # p_sub_features = torch.cat((m_features_1, m_features_2), axis = 0)
                # p_sub_label = torch.cat((m_pseudo_max, m_pseudo_max), axis = 0)

                # p_sub_class_1 = p_sub_features[torch.where(p_sub_label == 0)].mean(axis = 0)
                # p_sub_class_2 = p_sub_features[torch.where(p_sub_label == 1)].mean(axis = 0)  
                # p_sub_class_3 = p_sub_features[torch.where(p_sub_label == 2)].mean(axis = 0) 
                # p_sub_class_4 = p_sub_features[torch.where(p_sub_label == 3)].mean(axis = 0) 

                # r_sub_class_1 = d_features[torch.where(m_label_max == 0)].mean(axis = 0)
                # r_sub_class_2 = d_features[torch.where(m_label_max == 1)].mean(axis = 0)
                # r_sub_class_3 = d_features[torch.where(m_label_max == 2)].mean(axis = 0)
                # r_sub_class_4 = d_features[torch.where(m_label_max == 3)].mean(axis = 0)

                # constrastive_loss = functions.ContrastiveLoss()
                # con_val = constrastive_loss(r_target_class_1, r_sub_class_1, 0) + constrastive_loss(r_target_class_2, r_sub_class_2, 0) + \
                #             constrastive_loss(r_target_class_3, r_sub_class_3, 0) + constrastive_loss(r_target_class_4, r_sub_class_4, 0)
                
                # real_con = constrastive_loss(r_target_class_1, r_sub_class_1, 0) + constrastive_loss(r_target_class_1, r_sub_class_2, 1) + \
                #             constrastive_loss(r_target_class_1, r_sub_class_3, 1) + constrastive_loss(r_target_class_1, r_sub_class_4, 1) + \
                #             constrastive_loss(r_target_class_2, r_sub_class_2, 0) + constrastive_loss(r_target_class_2, r_sub_class_3, 1) + \
                #             constrastive_loss(r_target_class_2, r_sub_class_4, 1) + constrastive_loss(r_target_class_3, r_sub_class_3, 0) + \
                #             constrastive_loss(r_target_class_3, r_sub_class_4, 1) + constrastive_loss(r_target_class_4, r_sub_class_4, 0)
                        
                # adapt_con = constrastive_loss(r_target_class_1, p_sub_class_1, 0) + constrastive_loss(r_target_class_2, p_sub_class_2, 0) + \
                #             constrastive_loss(r_target_class_3, p_sub_class_3, 0) + constrastive_loss(r_target_class_4, p_sub_class_4, 0)
                
                # adapt_con = constrastive_loss(p_target_class_1, r_sub_class_1, 0) + constrastive_loss(p_target_class_2, r_sub_class_2, 0) + \
                #             constrastive_loss(p_target_class_3, r_sub_class_3, 0) + constrastive_loss(p_target_class_4, r_sub_class_4, 0)

                
                # intra_con = constrastive_loss(r_target_class_1, r_target_class_2, 1) + constrastive_loss(r_target_class_1, r_target_class_3, 1) + \
                #             constrastive_loss(r_target_class_1, r_target_class_4, 1) + constrastive_loss(r_target_class_2, r_target_class_3, 1) + \
                #             constrastive_loss(r_target_class_2, r_target_class_4, 1) + constrastive_loss(r_target_class_3, r_target_class_4, 1)    

                # with torch.no_grad():
                #     t_pseudo_label_2 = functions.sharpen(m_pseudo, 0.2).to(device)

                # t_cost_1_emax = F.mse_loss(m_predictions_1, t_pseudo_label_2).to(device)
                # t_cost_2_emax = F.mse_loss(m_predictions_2, t_pseudo_label_2).to(device)   
                # t_cost_3_emax = F.mse_loss(m_predictions_3, t_pseudo_label_2).to(device)

                # t_cost_emax = t_cost_1_emax + t_cost_2_emax + t_cost_3_emax
                m_domain_img, _ = target_samples
                m_domain_img = m_domain_img.to(device)
                m_2 = transformation.augment(m_domain_img, device)
                m_3 = transformation.augment(m_domain_img, device)

                m_domain_img = m_domain_img.unsqueeze(1)
                m_2 = m_2.unsqueeze(1)
                m_3 = m_3.unsqueeze(1)

                m_features_1 = encoder(m_domain_img)
                m_predictions_1 = cls_classifier(m_features_1, e_max = True)

                m_features_2 = encoder(m_2)
                m_predictions_2 = cls_classifier(m_features_2, e_max = True)

                m_features_3 = encoder(m_3)
                m_predictions_3 = cls_classifier(m_features_3, e_max = True)

                m_ent_1 = Categorical(probs = m_predictions_1).entropy().sum(axis = 0)
                m_ent_2 = Categorical(probs = m_predictions_2).entropy().sum(axis = 0)
                m_ent_3 = Categorical(probs = m_predictions_3).entropy().sum(axis = 0)

                max_ent = -(m_ent_1 + m_ent_2 + m_ent_3)
                # con_loss = mmd_loss + real_con + 0.8 * (epoch / nb_epochs) * adapt_con + 0.8 * intra_con
                con_loss = labeled_cost + dl_cost + 5 * max_ent
                
                optimizer.zero_grad()
                con_loss.requires_grad_(True)
                con_loss.backward()
                optimizer.step()
                
                for param in encoder.parameters():
                    param.requires_grad = True
            scheduler.step()
            
            if (epoch + 1 + (nb_epochs // 4)) % 50 == 0:
                torch.save(encoder.state_dict(), f'/home/workspace/Model/{args.name_exp}_{args.size_LD}_{random_seed}/{epoch + 1}_encoder.pth')
                torch.save(cls_classifier.state_dict(), f'/home/workspace/Model/{args.name_exp}_{args.size_LD}_{random_seed}/{epoch + 1}_classifier.pth')

            if (epoch+1) % 10 == 0:
                encoder.eval()
                cls_classifier.eval()
                with torch.no_grad():
                    
                    test_loss = 0.0
                    correct = 0
                    m_test_loss = 0.0
                    m_correct = 0

                    class_correct = list(0. for i in range(4))
                    class_total = list(0. for i in range(4))
                    for images, labels in valid_data_loader:
                        images = images.to(device)
                        images = images.unsqueeze(1)
                        labels = labels.to(device)
                        
                        features = encoder(images)
                        outputs = cls_classifier(features)

                        cost = F.cross_entropy(outputs, labels)
                        predicted = torch.max(outputs, 1)[1]
                        labels = torch.max(labels, 1)[1]
                        correct += (labels == predicted).sum()
                        test_loss += cost
                        c = (predicted == labels).squeeze()
                        for i in range(len(labels)):
                            class_correct[labels[i]] += c[i].item()
                            class_total[labels[i]] += 1
                    
                    m_class_correct = list(0. for i in range(4))
                    m_class_total = list(0. for i in range(4))
                    
                    for images, labels in multi_valid_data_loader:
                        images = images.to(device)
                        images = images.unsqueeze(1)
                        labels = labels.to(device)
                        
                        features = encoder(images)
                        outputs = cls_classifier(features)

                        cost = F.cross_entropy(outputs, labels)
                        predicted = torch.max(outputs, 1)[1]
                        labels = torch.max(labels, 1)[1]
                        m_correct += (labels == predicted).sum()
                        m_test_loss += cost
                        c = (predicted == labels).squeeze()
                        for i in range(len(labels)):
                            m_class_correct[labels[i]] += c[i].item()
                            m_class_total[labels[i]] += 1
                print('--------------- validation stage -----------------')
                total = class_total[0] + class_total[1] + class_total[2] + class_total[3]  
                print (f'epoch{epoch + 1}, cls_loss: {test_loss}, acc: {(correct / total)}')
                print(f'class_0 {class_correct[0] / class_total[0]}')
                try:
                    print(f'class_1 {class_correct[1] / class_total[1]}')
                except:
                    print('No Ball bearing fault')
                print(f'class_2 {class_correct[2] / class_total[2]}')
                print(f'class_3 {class_correct[3] / class_total[3]}')
                print('mmd_loss',mmd_loss)
                
                m_total = m_class_total[0] + m_class_total[1] + m_class_total[2] + m_class_total[3]
                print('--------------- multi domain validation stage -----------------')  
                print (f'epoch{epoch + 1}, cls_loss: {m_test_loss}, acc: {(m_correct / m_total)}')
                print(f'class_0 {m_class_correct[0] / m_class_total[0]}')
                try:
                    print(f'class_1 {m_class_correct[1] / m_class_total[1]}')
                except:
                    print('No ball bearing fault')
                print(f'class_2 {m_class_correct[2] / m_class_total[2]}')
                print(f'class_3 {m_class_correct[3] / m_class_total[3]}')
                #print('contrastive loss', con_val)
                print('entory', max_ent)
                acc.append((correct / total))
                early_stopping(test_loss, encoder, cls_classifier)

                if early_stopping.early_stop:
                    print("Early stopping")
                    break
        
    encoder.eval()
    cls_classifier.eval()
    with torch.no_grad():
        
        test_loss = 0.0
        correct = 0

        class_correct = list(0. for i in range(4))
        class_total = list(0. for i in range(4))
        for images, labels in test_dataloader:
            images = images.to(device)
            images = images.unsqueeze(1)
            labels = labels.to(device)
            
            features = encoder(images)
            predicted = cls_classifier(features)

            cost = F.cross_entropy(predicted, labels)
            predicted = torch.max(predicted, 1)[1]
            labels = torch.max(labels, 1)[1]
            correct += (labels == predicted).sum()
            test_loss += cost
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                class_correct[labels[i]] += c[i].item()
                class_total[labels[i]] += 1
    total = class_total[0] + class_total[1] + class_total[2] + class_total[3]  
    print('--------------- Test Stage -----------------')
    print (f'cls_loss: {test_loss} acc: {(correct / total)}')
    print(f'class_0 {class_correct[0] / class_total[0]}')
    try:
        print(f'class_1 {class_correct[1] / class_total[1]}')
    except:
        print('No Ball bearing fault')
    print(f'class_2 {class_correct[2] / class_total[2]}')
    print(f'class_3 {class_correct[3] / class_total[3]}')
    acc.append((correct / total))
    
    for i in range(len(acc)):
        acc[i] = acc[i].cpu()
    
    torch.save(encoder.state_dict(), f'/home/workspace//Model/{args.name_exp}_{args.size_LD}_{random_seed}/encoder.pth')
    torch.save(cls_classifier.state_dict(), f'/home/workspace/Model/{args.name_exp}_{args.size_LD}_{random_seed}/cls_classifier.pth')
    np.save(f'/home/workspace/Model/{args.name_exp}_{args.size_LD}_{random_seed}/acc.npy', np.array(acc))