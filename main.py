import argparse
import warnings
import sklearn.exceptions


import trainers

warnings.filterwarnings("ignore", category=sklearn.exceptions.UndefinedMetricWarning)


parser = argparse.ArgumentParser()


# ========  Experiments Name ================
parser.add_argument('--save_dir', default='logs', type=str, help='Directory containing all experiments')
parser.add_argument('--experiment_description', default='OTDA', type=str, help='Name of your experiment (UCIHAR, HHAR_P, WISDM)')
parser.add_argument('--run_description', default='UCIHAR', type=str, help='name of your runs')

# ========= Select the DA methods ============
parser.add_argument('--da_method', default='OTDA', type=str)

# ========= Select the DATASET ==============
# parser.add_argument('--data_path', default='D:\\BigBang\\data', type=str, help='Path containing dataset')
parser.add_argument('--data_path', default='/data/liushubin', type=str, help='Path containing dataset')
parser.add_argument('--dataset', default='WISDM',type=str)

# ========= Select the BACKBONE ==============
parser.add_argument('--backbone', default='CNN', type=str)

# ========= Experiment settings ===============
parser.add_argument('--num_runs', default=5, type=int, help='Number of consecutive run with different seeds')
parser.add_argument('--device', default='cuda', type=str, help='cpu or cuda')
parser.add_argument('--num_epochs', type=int, default=50)
parser.add_argument('--bs', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.01, help='optimizer learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-4)
parser.add_argument('--start',type=int, default=0)
parser.add_argument('--end', type=int, default=None)
parser.add_argument('-p','--print-freq', type=int, default=10, help='each epoch print num_epochs/p times ')
parser.add_argument('--num_workers', type=int, default=2)
parser.add_argument('--shuffle', action='store_true', help='whether shuffle the train dataset')
parser.add_argument('--phase', default='train', type=str)
parser.add_argument('--test_model_prefix', type=str)

# =========        ACON       ===============
parser.add_argument('--kl_reduction', default='mean')
parser.add_argument('--kl_t',default=1.0, type=float)
parser.add_argument('--disc_hid_dim', type=int, default=128)
# trade_off for different loss
parser.add_argument('--entropy_trade_off', type=float,default=0.01)
parser.add_argument('--domain_trade_off', type=float,default=1.0)
parser.add_argument('--align_s_trade_off', type=float,default=1.0)
parser.add_argument('--align_t_trade_off', type=float,default=1.0)
parser.add_argument('--cls_trade_off', type=float,default=1.0)

# ========= Iteration & Test settings ============
parser.add_argument('--stop_step', type=int, default=500, help='Total number of iterations')
parser.add_argument('--test_interval', type=int, default=5, help='Test accuracy every N steps')
parser.add_argument('--print_freq', type=int, default=1)

# =========   Mini-batch OT Parameters   ===============
parser.add_argument("--ot_type", type=str, default="balanced", choices=["balanced", "unbalanced", "partial"], help="Type of optimal transport")
parser.add_argument("--eta1", type=float, default=0.01, help="weight of embedding loss (Euclidean distance)")
parser.add_argument("--eta2", type=float, default=0.5, help="weight of transportation loss (Softmax Cross Entropy)")
parser.add_argument("--epsilon", type=float, default=0.5, help="OT regularization coefficient (Sinkhorn entropy)")
parser.add_argument("--tau", type=float, default=0.3, help="marginal penalization coefficient for unbalanced OT")


parser.add_argument("--mass", type=float, default=0.7, help="ratio of masses to be transported for partial OT")
parser.add_argument("--ot_t_trade_off", type=float, default=0.5, help="Weight for the time OT loss")
parser.add_argument("--ot_f_trade_off", type=float, default=0.5, help="Weight for the frequency OT loss")
parser.add_argument("--k", type=int, default=1, help="number of minibatches to average over")
parser.add_argument("--topk", type=int, default=6, help="number of main patterns to compose the feature representation")

parser.add_argument('--use_balanced_sampler', type=int, default=1, choices=[0, 1],
                    help='Whether to use BalancedBatchSampler for the source domain (1 for yes, 0 for no. Default: 1)')
parser.add_argument("--freq_weight_lambda", type=float, default=0, help="frequency weight lambda")
parser.add_argument("--time_weight_lambda", type=float, default=0, help="time weight lambda")
parser.add_argument("--mix_lambda", type=float, default=0, help="mix_lambda")
parser.add_argument("--freq_aux_trade_off", type=float, default=0, help="frequency auxiliary trade off")
parser.add_argument("--marginal_smooth", type=float, default=1e-6, help="marginal smoothness parameter")



# ========= t-SNE visualization settings ============
parser.add_argument('--src_id', type=str, default='5', help='Source domain id for t-SNE visualization')
parser.add_argument('--trg_id', type=str, default='1', help='Target domain id for t-SNE visualization')
parser.add_argument('--run_id', type=int, default=0, help='Run id for t-SNE visualization')
parser.add_argument('--model_path', type=str, default=None, help='Path to trained model.pth for t-SNE visualization')
parser.add_argument('--tsne_save_dir', type=str, default='tsne_results', help='Directory to save t-SNE figures')
parser.add_argument('--max_points_per_domain', type=int, default=None, help='Maximum number of points per domain for t-SNE')



args = parser.parse_args()



if __name__ == "__main__":
    
    trainer = trainers.da_trainer(args)
    if args.phase == 'train':
        trainer.train()

    elif args.phase == 'test':
        trainer.test()

    elif args.phase == 'tsne':
        trainer.visualize_tsne(
            src_id=args.src_id,
            trg_id=args.trg_id,
            run_id=args.run_id,
            model_path=args.model_path,
            save_dir=args.tsne_save_dir,
            max_points_per_domain=args.max_points_per_domain
        )

    else:
        raise ValueError(f"Unknown phase: {args.phase}")
   