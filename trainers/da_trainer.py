
import torch
import os
import sys
import pandas as pd
import numpy as np
import warnings
import sklearn.exceptions
warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)
import collections


from sklearn.metrics import accuracy_score
from tqdm import tqdm
from dataloader.dataloader import data_generator
from configs.data_model_configs import get_dataset_class
from algorithms.utils import fix_randomness, starting_logs
from algorithms import get_algorithm_class


from algorithms.utils import get_time
from algorithms.utils import AverageMeter
from sklearn.metrics import f1_score

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

torch.backends.cudnn.benchmark = True  
warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)        
   

class da_trainer(object):
    """
   This class contain the main training functions for our AdAtime
    """
    def __init__(self, args):
        self.args = args
        self.da_method = args.da_method  # Selected  DA Method
        self.dataset = args.dataset  # Selected  Dataset
        self.backbone = args.backbone
        self.device = torch.device(args.device)  # device

        self.run_description = args.run_description
        self.experiment_description = args.experiment_description

        self.best_f1 = 0
        # paths
        self.home_path = os.getcwd()
        self.save_dir = args.save_dir
        self.data_path = os.path.join(args.data_path, self.dataset)
        self.create_save_dir()

        # Specify runs
        self.num_runs = args.num_runs

        # get dataset and base model configs
        self.dataset_configs = self.get_configs()
       
        # to fix dimension of features in classifier and discriminator networks.
        self.dataset_configs.t_feat_dim = self.dataset_configs.tcn_final_out_channles if args.backbone == "TCN" else self.dataset_configs.t_feat_dim
        self.args.seq_len = self.dataset_configs.sequence_len
        self.args.enc_in = self.dataset_configs.input_channels
        
    # 貌似没啥用这个test函数，self.args.test_model_prefix未指定，并且相应的逻辑在train函数体现了
    def test(self):
    
        run_name = f"{self.run_description}"
        # Logging
        self.avg_res_dir = os.path.join(self.save_dir, self.experiment_description, run_name, get_time())
        os.makedirs(self.avg_res_dir, exist_ok=True)

        self.exp_log_dir = os.path.join(self.avg_res_dir, 'res')
        os.makedirs(self.exp_log_dir, exist_ok=True)

        command = ' '.join(sys.argv)
        with open(os.path.join(self.avg_res_dir, 'command.txt'), "a") as file:
            command_list = command.split('--')
            for arg in command_list:
                file.write('--'+arg+'\n')
                file.flush()


        scenarios = self.dataset_configs.scenarios  # return the scenarios given a specific dataset.
        df_a = pd.DataFrame(columns=['scenario','run_id','accuracy','f1'])
        df_s = pd.DataFrame(columns=['scenario','run_id','accuracy','f1'])
        self.trg_acc_list = []
        for i in scenarios[self.args.start:self.args.end]:
            src_id = i[0]
            trg_id = i[1]


            for run_id in range(self.num_runs):  # specify number of consecutive runs
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                   src_id, trg_id, run_id)
                # Load data
                self.load_data(src_id, trg_id)
                # get algorithm
                print(self.da_method)
                algorithm_class = get_algorithm_class(self.da_method)
                algorithm = algorithm_class(self.dataset_configs, self.device, self.args)
                
                algorithm.to(self.device)
                self.algorithm = algorithm       
                self.logger.debug('Source Test Dataset {}  Target Test Dataset {}'.format(len(self.src_test_dl), len(self.trg_test_dl)))

                self.algorithm.load_model(os.path.join(self.args.test_model_prefix, 'res', f'{src_id}_to_{trg_id}_run_{run_id}', 'model.pth'))
                # test target
                acc, f1 = self.evaluate()
                log = {'scenario':i,'run_id':run_id,'accuracy':acc,'f1':f1}
                self.logger.debug('target acc {} f1 {}'.format(acc, f1))
                df_a = pd.concat([df_a, pd.DataFrame([log])], ignore_index=True)
                
                # test source
                acc, f1 = self.evaluate(data='s')
                log = {'scenario':i,'run_id':run_id,'accuracy':acc,'f1':f1}
                self.logger.debug('source acc {} f1 {}'.format(acc, f1))
                df_s = pd.concat([df_s, pd.DataFrame([log])], ignore_index=True)
                
                path =  os.path.join(self.avg_res_dir, 'test_target_results.csv')
                df_a.to_csv(path,sep = ',')
                path_s =  os.path.join(self.avg_res_dir, 'test_source_results.csv')
                df_s.to_csv(path_s,sep = ',')
       
        df_a = self.avg_result(df_a)
        df_s = self.avg_result(df_s)

        path =  os.path.join(self.avg_res_dir, 'test_target_results.csv')
        df_a.to_csv(path,sep = ',')
        path_s =  os.path.join(self.avg_res_dir, 'test_source_results.csv')
        df_s.to_csv(path_s,sep = ',')
        


            
    def train(self):

        run_name = f"{self.run_description}"
        # Logging
        self.avg_res_dir = os.path.join(self.save_dir, self.experiment_description, run_name, get_time())
        os.makedirs(self.avg_res_dir, exist_ok=True)

        self.exp_log_dir = os.path.join(self.avg_res_dir, 'res')
        os.makedirs(self.exp_log_dir, exist_ok=True)

        command = ' '.join(sys.argv)
        with open(os.path.join(self.avg_res_dir, 'command.txt'), "a") as file:
            command_list = command.split('--')
            for arg in command_list:
                file.write('--'+arg+'\n')
                file.flush()

        scenarios = self.dataset_configs.scenarios  # return the scenarios given a specific dataset.
        df_a = pd.DataFrame(columns=['scenario','run_id','accuracy','f1'])
        df_s = pd.DataFrame(columns=['scenario','run_id','accuracy','f1'])
        
        for i in scenarios[self.args.start:self.args.end]:
            src_id = i[0]
            trg_id = i[1]


            for run_id in range(self.num_runs):  # specify number of consecutive runs
                # fixing random seed
                fix_randomness(run_id)

                # Logging
                self.logger, self.scenario_log_dir = starting_logs(self.dataset, self.da_method, self.exp_log_dir,
                                                                   src_id, trg_id, run_id)
                self.model_path = os.path.join(self.home_path, self.scenario_log_dir, 'model.pth')
                # Load data
                self.load_data(src_id, trg_id)

    
                # get algorithm
                print(self.da_method)
               
                algorithm_class = get_algorithm_class(self.da_method)
                algorithm = algorithm_class(self.dataset_configs, self.device, self.args)
                
                algorithm.to(self.device)
                self.algorithm = algorithm

                # Training variables
                self.best_f1 = 0
                self.best_acc = 0

                # Manually create iterators
                iter_source = iter(self.src_train_dl)
                iter_target = iter(self.trg_train_dl)

                # Average meters
                loss_avg_meters = collections.defaultdict(lambda: AverageMeter())
                self.logger.debug('Source Train Dataset {}  Target Train Dataset {}'.format(len(self.src_train_dl), len(self.trg_train_dl)))

                # =============================================
                # Main Training Loop (Iterations based)
                # =============================================
                for step in tqdm(range(1, self.args.stop_step + 1), desc=f"Run {run_id} {src_id}->{trg_id}"):
                    
                    # 1. Test Step
                    if step % self.args.test_interval == 0 and step > 0:
                        self.logger.debug('Step Testing {}/{}'.format(step, self.args.stop_step))

                        acc, f1 = self.evaluate(data='t')
                        self.logger.debug('acc {}   f1 {}'.format(acc, f1))

                        if f1 >= self.best_f1:
                            self.best_f1 = f1
                            self.logger.debug('best model at step {}'.format(step))
                            self.algorithm.save_model(self.model_path)

                    # 2. Train Step with k-batch Averaging
                    # Reset Gradients BEFORE the k-loop
                    self.algorithm.train()
                    self.algorithm.optimizer.zero_grad()

                    for _ in range(self.args.k):
                        # --- Try-Except Data Loading ---
                        try:
                            xs_mb, ys_mb = next(iter_source)
                        except StopIteration:
                            iter_source = iter(self.src_train_dl)
                            xs_mb, ys_mb = next(iter_source)
                        
                        try:
                            xt_mb, _ = next(iter_target)
                        except StopIteration:
                            iter_target = iter(self.trg_train_dl)
                            xt_mb, _ = next(iter_target)
                        
                        xs_mb = xs_mb.float().to(self.device)
                        ys_mb = ys_mb.long().to(self.device)
                        xt_mb = xt_mb.float().to(self.device)

                        # --- Update (Backward only, accumulate gradient) ---
                        # apply_step=False means we only do backward, not step
                        losses = algorithm.update(xs_mb, ys_mb, xt_mb, apply_step=False)

                        # Accumulate losses for logging
                        for key, val in losses.items():
                            loss_avg_meters[key].update(val, self.args.bs)

                    # 3. Apply Optimizer Step AFTER k batches
                    self.algorithm.optimizer.step()
                    
                    # Logging
                    if step % self.args.print_freq == 0:
                        keys = loss_avg_meters.keys()
                        train_log = 'step {}   '.format(step)
                        for key in keys:
                            train_log += '{}    {:.3f}({:.3f})    '.format(key,loss_avg_meters[key].val, loss_avg_meters[key].avg)

                        self.logger.debug(train_log)

                # =============================================
                # End of Run Evaluation
                # =============================================
                
                # 1. Test on Target
                acc, f1 = self.evaluate(final=True)
                log = {'scenario':i,'run_id':run_id,'accuracy':acc,'f1':f1}
                self.logger.debug('target acc {} f1 {}'.format(acc, f1))
                df_a = pd.concat([df_a, pd.DataFrame([log])], ignore_index=True)

                # 2. Test on Source
                acc, f1 = self.evaluate(final=True, data='s')
                log = {'scenario':i,'run_id':run_id,'accuracy':acc,'f1':f1}
                self.logger.debug('source acc {} f1 {}'.format(acc, f1))
                df_s = pd.concat([df_s, pd.DataFrame([log])], ignore_index=True)

                path =  os.path.join(self.avg_res_dir, 'target_results.csv')
                df_a.to_csv(path,sep = ',')
                path_s =  os.path.join(self.avg_res_dir, 'source_results.csv')
                df_s.to_csv(path_s,sep = ',')

        
        df_a = self.avg_result(df_a)
        df_s = self.avg_result(df_s)

        path =  os.path.join(self.avg_res_dir, 'target_results.csv')
        df_a.to_csv(path,sep = ',')
        path_s =  os.path.join(self.avg_res_dir, 'source_results.csv')
        df_s.to_csv(path_s,sep = ',')

    
    def evaluate(self, final=False, data='t'):
        assert data in ['t', 's']
        self.algorithm.eval()
        if final == True:
            self.algorithm.load_model(self.model_path)
    
        self.trg_pred_labels = np.array([])
        self.trg_true_labels = np.array([])

        if data == 't':
            dataloader = self.trg_test_dl
        elif data == 's':
            dataloader = self.src_test_dl

        with torch.no_grad():
            for data, labels in dataloader:
                data = data.float().to(self.device)
                labels = labels.view((-1)).long().to(self.device)
                predictions = self.algorithm.predict(data)
                # compute loss
                pred = predictions.detach().argmax(dim=1)  # get the index of the max log-probability

                self.trg_pred_labels = np.append(self.trg_pred_labels, pred.cpu().numpy())
                self.trg_true_labels = np.append(self.trg_true_labels, labels.data.cpu().numpy())
        
        accuracy = accuracy_score(self.trg_true_labels, self.trg_pred_labels)
        f1 = f1_score(self.trg_true_labels, self.trg_pred_labels, pos_label=None, average="macro")
        # f1 = f1_score(self.trg_pred_labels, self.trg_true_labels, pos_label=None, average="macro")
        return accuracy*100, f1


    def get_configs(self):
        dataset_class = get_dataset_class(self.dataset)
        return dataset_class()

    def load_data(self, src_id, trg_id):
        self.src_train_dl, self.src_test_dl = data_generator(self.data_path, src_id, self.args, is_src=True)
        self.trg_train_dl, self.trg_test_dl = data_generator(self.data_path, trg_id, self.args, is_src=False)
    

    def create_save_dir(self):
        if not os.path.exists(self.save_dir):       
            os.mkdir(self.save_dir)

    def avg_result(self, df):

        empty_row = [{'scenario': None, 'run_id': None, 'accuracy': None, 'f1': None}]
        df = pd.concat([df, pd.DataFrame(empty_row)], ignore_index=True)

        mean_acc = df.groupby('scenario', as_index=False, sort=False)['accuracy'].mean(numeric_only=True)
        mean_f1 = df.groupby('scenario', as_index=False, sort=False)['f1'].mean(numeric_only=True)
        std_acc = df.groupby('scenario', as_index=False, sort=False)['accuracy'].std(numeric_only=True)
        std_f1 =  df.groupby('scenario', as_index=False, sort=False)['f1'].std(numeric_only=True)

        print(mean_acc)
        print(std_acc)

        for i in range(len(mean_acc)):
            log = [{'scenario':mean_acc['scenario'][i],'run_id':'all','accuracy':mean_acc['accuracy'][i],'f1':mean_f1['f1'][i]}]
            log.append({'scenario':mean_acc['scenario'][i],'run_id':'all','accuracy':std_acc['accuracy'][i],'f1':std_f1['f1'][i]})
            df = pd.concat([df, pd.DataFrame(log)], ignore_index=True)
        
        all_mean_acc = mean_acc['accuracy'].mean()
        all_mean_f1 = mean_f1['f1'].mean()
        all_mean_acc_std = std_acc['accuracy'].mean()
        all_mean_f1_std = std_f1['f1'].mean()
        log = [
            {'scenario':'all_mean_acc',
                'run_id':'all_mean_acc_std',
                'accuracy':'all_mean_f1',
                'f1':'all_mean_f1_std'},
            {'scenario':all_mean_acc,
                'run_id':all_mean_acc_std,
                'accuracy':all_mean_f1,
                'f1':all_mean_f1_std}]
        
        df = pd.concat([df, pd.DataFrame(log)], ignore_index=True)

        return df
    
    def collect_tsne_features(self, dataloader, domain_id):
        """
        Extract temporal and frequency features from a dataloader.

        Args:
            dataloader: source or target test dataloader
            domain_id: 0 for source domain, 1 for target domain

        Returns:
            t_features: temporal features, shape [N, D_t]
            f_features: frequency features, shape [N, D_f]
            labels: class labels, shape [N]
            domains: domain labels, shape [N]
        """
        self.algorithm.eval()

        t_features_list = []
        f_features_list = []
        labels_list = []
        domains_list = []

        with torch.no_grad():
            for data, labels in dataloader:
                data = data.float().to(self.device)
                labels = labels.view(-1).long()

                feat_t, feat_f = self.algorithm.extract_for_tsne(data)

                t_features_list.append(feat_t.detach().cpu().numpy())
                f_features_list.append(feat_f.detach().cpu().numpy())
                labels_list.append(labels.cpu().numpy())
                domains_list.append(
                    np.full(labels.size(0), domain_id, dtype=np.int64)
                )

        t_features = np.concatenate(t_features_list, axis=0)
        f_features = np.concatenate(f_features_list, axis=0)
        labels = np.concatenate(labels_list, axis=0)
        domains = np.concatenate(domains_list, axis=0)

        return t_features, f_features, labels, domains

    def plot_tsne_domain_class(
        self,
        features,
        labels,
        domains,
        save_path,
        title="t-SNE",
        num_classes=6,
        random_state=42,
        max_points_per_domain=None
    ):
        """
        Draw t-SNE visualization.

        Color indicates class label.
        Marker indicates domain:
            Source: circle
            Target: triangle
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Optional: subsample points to avoid overcrowded t-SNE
        if max_points_per_domain is not None:
            selected_indices = []

            for d in [0, 1]:
                domain_idx = np.where(domains == d)[0]
                if len(domain_idx) > max_points_per_domain:
                    rng = np.random.default_rng(random_state + d)
                    domain_idx = rng.choice(
                        domain_idx,
                        size=max_points_per_domain,
                        replace=False
                    )
                selected_indices.append(domain_idx)

            selected_indices = np.concatenate(selected_indices)
            features = features[selected_indices]
            labels = labels[selected_indices]
            domains = domains[selected_indices]

        tsne = TSNE(
            n_components=2,
            perplexity=30,
            learning_rate="auto",
            init="pca",
            random_state=random_state
        )

        features_2d = tsne.fit_transform(features)

        plt.rcParams.update({
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.unicode_minus": False,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
        })

        fig, ax = plt.subplots(figsize=(5.2, 4.6))

        # 6 class colors, pastel but distinguishable
        colors = [
            "#8EC7E6",  # Class 0, light blue
            "#F3A6A6",  # Class 1, light red
            "#A8D5BA",  # Class 2, light green
            "#D6C6F2",  # Class 3, light purple
            "#F6D58B",  # Class 4, light yellow
            "#B8B8B8",  # Class 5, light gray
        ]

        markers = {
            0: "o",   # Source
            1: "^",   # Target
        }

        domain_names = {
            0: "Source",
            1: "Target",
        }

        for c in range(num_classes):
            for d in [0, 1]:
                mask = (labels == c) & (domains == d)

                if np.sum(mask) == 0:
                    continue

                ax.scatter(
                    features_2d[mask, 0],
                    features_2d[mask, 1],
                    c=colors[c % len(colors)],
                    marker=markers[d],
                    s=24,
                    alpha=0.78,
                    edgecolors="white",
                    linewidths=0.25,
                )

        ax.set_title(title, pad=8, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

        for spine in ax.spines.values():
            spine.set_linewidth(0.9)

        # Legend 1: class colors
        class_handles = [
            Line2D(
                [0], [0],
                marker="o",
                color="none",
                markerfacecolor=colors[c % len(colors)],
                markeredgecolor="white",
                markersize=7,
                label=f"Class {c}"
            )
            for c in range(num_classes)
        ]

        # Legend 2: domain markers
        domain_handles = [
            Line2D(
                [0], [0],
                marker="o",
                color="gray",
                linestyle="none",
                markersize=7,
                label="Source"
            ),
            Line2D(
                [0], [0],
                marker="^",
                color="gray",
                linestyle="none",
                markersize=7,
                label="Target"
            )
        ]

        legend1 = ax.legend(
            handles=class_handles,
            loc="upper right",
            frameon=False,
            title="Class",
            fontsize=8,
            title_fontsize=9
        )
        ax.add_artist(legend1)

        ax.legend(
            handles=domain_handles,
            loc="lower right",
            frameon=False,
            title="Domain",
            fontsize=8,
            title_fontsize=9
        )

        plt.tight_layout(pad=0.4)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()
        # plt.close()
    
    
    
    
    def visualize_tsne(
        self,
        src_id=5,
        trg_id=1,
        run_id=0,
        model_path=None,
        save_dir=None,
        max_points_per_domain=None
    ):
        """
        Visualize temporal and frequency features on source and target test sets.

        Args:
            src_id: source domain id
            trg_id: target domain id
            run_id: random seed / run id
            model_path: path to trained model.pth
            save_dir: directory to save t-SNE figures
            max_points_per_domain: optional subsampling number for each domain
        """
        fix_randomness(run_id)

        # 1. Load source and target test data
        self.load_data(src_id, trg_id)

        # 2. Build algorithm
        algorithm_class = get_algorithm_class(self.da_method)
        algorithm = algorithm_class(self.dataset_configs, self.device, self.args)
        algorithm.to(self.device)
        self.algorithm = algorithm

        # 3. Load trained model
        if model_path is None:
            model_path = os.path.join(
                self.save_dir,
                self.experiment_description,
                self.run_description,
                "res",
                f"{src_id}_to_{trg_id}_run_{run_id}",
                "model.pth"
            )

        print(f"Loading model from: {model_path}")
        self.algorithm.load_model(model_path)

        # 4. Extract source test features
        src_t, src_f, src_y, src_d = self.collect_tsne_features(
            self.src_test_dl,
            domain_id=0
        )

        # 5. Extract target test features
        trg_t, trg_f, trg_y, trg_d = self.collect_tsne_features(
            self.trg_test_dl,
            domain_id=1
        )

        # 6. Concatenate source and target features
        all_t = np.concatenate([src_t, trg_t], axis=0)
        all_f = np.concatenate([src_f, trg_f], axis=0)
        all_y = np.concatenate([src_y, trg_y], axis=0)
        all_d = np.concatenate([src_d, trg_d], axis=0)

        print("Temporal feature shape:", all_t.shape)
        print("Frequency feature shape:", all_f.shape)
        print("Label shape:", all_y.shape)
        print("Domain shape:", all_d.shape)

        # 7. Set save directory
        if save_dir is None:
            save_dir = os.path.join(
                self.save_dir,
                "tsne",
                self.dataset,
                f"{src_id}_to_{trg_id}_run_{run_id}"
            )

        os.makedirs(save_dir, exist_ok=True)

        # 8. Plot temporal feature t-SNE
        self.plot_tsne_domain_class(
            features=all_t,
            labels=all_y,
            domains=all_d,
            save_path=os.path.join(save_dir, "tsne_temporal.png"),
            title="Temporal Feature t-SNE",
            num_classes=self.dataset_configs.num_classes,
            random_state=run_id,
            max_points_per_domain=max_points_per_domain
        )

        # 9. Plot frequency feature t-SNE
        self.plot_tsne_domain_class(
            features=all_f,
            labels=all_y,
            domains=all_d,
            save_path=os.path.join(save_dir, "tsne_frequency.png"),
            title="Frequency Feature t-SNE",
            num_classes=self.dataset_configs.num_classes,
            random_state=run_id,
            max_points_per_domain=max_points_per_domain
        )

        print(f"t-SNE figures saved to: {save_dir}")