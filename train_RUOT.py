import os
import sys
import argparse
import pandas as pd
import torch
import anndata as ad
from tqdm import tqdm
import random
import numpy as np

# set random seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.random.manual_seed(SEED)

#sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))

from DeepRUOT.losses import OT_loss1
from DeepRUOT.utils import (
    generate_steps, load_and_merge_config,
    SchrodingerBridgeConditionalFlowMatcher,
    generate_state_trajectory, get_batch, get_batch_size
)
from DeepRUOT.train import train_un1_reduce, train_all
from DeepRUOT.models import FNet, scoreNet2
from DeepRUOT.constants import DATA_DIR
from DeepRUOT.exp import setup_exp

class TrainingPipeline:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        self.exp_dir = None
        self.logger = None
        
        # Initialize experiment directory and logger
        self._setup_experiment()
        
        # Initialize models and data
        self.f_net = None
        self.sf2m_score_model = None
        self.df = None
        self.groups = None
        self.steps = None
        self.relative_mass = None
        self.initial_size = None
        
        self._setup_models()
        self._load_data()
        self._setup_training()

    def _setup_experiment(self):
        """Setup experiment directory and logger"""
        self.exp_dir, self.logger = setup_exp(
            self.config['exp']['output_dir'], 
            self.config, 
            self.config['exp']['name']
        )
        self.logger.info(f'Starting experiment in {self.exp_dir}')

    def _setup_models(self):
        """Initialize neural network models"""
        model_config = self.config['model']
        
        self.f_net = FNet(
            in_out_dim=model_config['in_out_dim'],
            hidden_dim=model_config['hidden_dim'],
            n_hiddens=model_config['n_hiddens'],
            activation=model_config['activation']
        ).to(self.device)
        
        self.sf2m_score_model = scoreNet2(
            in_out_dim=model_config['in_out_dim'],
            hidden_dim=model_config['score_hidden_dim'],
            activation=model_config['activation']
        ).float().to(self.device)

    def _load_data(self):
        """Load and preprocess data"""
        self.df = pd.read_csv(os.path.join(DATA_DIR, self.config['data']['file_path']))
        self.df = self.df.iloc[:, :self.config['data']['dim'] + 1]

    def _setup_training(self):
        """Setup training parameters"""
        self.groups = sorted(self.df.samples.unique())
        self.steps = generate_steps(self.groups)
        
        hold_one_out = self.config['data']['hold_one_out']
        hold_out = self.config['data']['hold_out']
        
        if hold_one_out:
            df_mass = self.df[self.df['samples'] != hold_out]
            sample_sizes = df_mass.groupby('samples').size()
        else:
            sample_sizes = self.df.groupby('samples').size()
        
        ref0 = sample_sizes / sample_sizes.iloc[0]
        self.relative_mass = torch.tensor(ref0.values, dtype=torch.float32)
        self.relative_mass = self.relative_mass.to(self.device)
        
        self.initial_size = self.df[self.df['samples']==0].x1.shape[0]

    def pretrain(self):
        """Phase 1: Pretrain"""
        self.logger.info('Pretraining growth net')
        optimizer = torch.optim.Adam(self.f_net.parameters(), self.config['pretrain']['lr'])
        criterion = OT_loss1(which='emd', device=self.device)
        sample_size = (self.config['sample_size'],)
        
        l_loss, b_loss, g_loss = train_un1_reduce(
            self.f_net, self.df, self.groups, optimizer, self.config['pretrain']['epochs'],
            criterion=criterion,
            local_loss=True,
            global_loss=False,
            apply_losses_in_time=self.config['apply_losses_in_time'],
            hold_one_out=self.config['data']['hold_one_out'],
            hold_out=self.config['data']['hold_out'],
            hinge_value=self.config['hinge_value'],
            lambda_ot=self.config['pretrain']['lambda_ot'],
            lambda_mass=self.config['pretrain']['lambda_mass'],
            lambda_energy=self.config['pretrain']['lambda_energy'],
            use_pinn=False,
            use_penalty=self.config['use_penalty'],
            use_density_loss=self.config['use_density_loss'],
            lambda_density=self.config['lambda_density'],
            top_k=self.config['top_k'],
            sample_size=sample_size,
            relative_mass=self.relative_mass,
            initial_size=self.initial_size,
            sample_with_replacement=self.config['sample_with_replacement'],
            device=self.device,
            best_model_path=os.path.join(self.exp_dir, 'best_model'),
            logger=self.logger
        )

        self.logger.info('Refining velocity')
        self.f_net.load_state_dict(torch.load(os.path.join(self.exp_dir, 'best_model'), map_location=self.device))
        for param in self.f_net.g_net.parameters():
            param.requires_grad = False
        optimizer = torch.optim.Adam(self.f_net.parameters(), 1e-5)
        criterion = OT_loss1(which='emd', device=self.device)
        sample_size = (self.config['sample_size'],)
        
        l_loss, b_loss, g_loss = train_un1_reduce(
            self.f_net, self.df, self.groups, optimizer, self.config['pretrain']['epochs']//5,
            criterion=criterion,
            local_loss=True,
            global_loss=False,
            apply_losses_in_time=self.config['apply_losses_in_time'],
            hold_one_out=self.config['data']['hold_one_out'],
            hold_out=self.config['data']['hold_out'],
            hinge_value=self.config['hinge_value'],
            lambda_ot=1,
            lambda_mass=0,
            lambda_energy=0,
            use_pinn=False,
            use_penalty=False,
            use_density_loss=self.config['use_density_loss'],
            lambda_density=self.config['lambda_density'],
            top_k=self.config['top_k'],
            sample_size=sample_size,
            relative_mass=self.relative_mass,
            initial_size=self.initial_size,
            sample_with_replacement=self.config['sample_with_replacement'],
            device=self.device,
            best_model_path=os.path.join(self.exp_dir, 'best_model'),
            logger=self.logger
        )
        
        return l_loss, b_loss, g_loss

    def train_score_model(self):
        """Phase 2: Train score model"""
        self.logger.info('Training score model')
        self.f_net.load_state_dict(torch.load(os.path.join(self.exp_dir, 'best_model'), map_location=self.device))
        
        # Prepare data
        dim = self.config['data']['dim']
        samples = self.df['samples'].values
        column_names = [f'x{i}' for i in range(1, dim + 1)]
        obsm_data = self.df[column_names].values
        
        adata = ad.AnnData(obs=pd.DataFrame(index=samples))
        adata.obsm['X_pca'] = obsm_data
        adata.obs['samples'] = samples
        
        n_times = len(adata.obs["samples"].unique())
        X = [adata.obsm["X_pca"][adata.obs["samples"] == t] for t in range(n_times)]
        
        # Setup training
        batch_size = self.config['sample_size']
        sigma = self.config['score_train']['sigma']
        time = torch.Tensor(self.groups)
        SF2M = SchrodingerBridgeConditionalFlowMatcher(sigma=sigma)
        
        sf2m_optimizer = torch.optim.AdamW(
            list(self.sf2m_score_model.parameters()),
            self.config['score_train']['lr']
        )
        
        trajectory = generate_state_trajectory(X, n_times, self.df[self.df['samples']==0].x1.shape[0], self.f_net, time, self.device)
        
        # Training loop
        for i in tqdm(range(self.config['score_train']['epochs']), desc='Training score model'):
            sf2m_optimizer.zero_grad()
            t, xt, ut, eps = get_batch_size(SF2M, X, trajectory, batch_size, n_times, return_noise=True)
            t = torch.unsqueeze(t, 1)
            lambda_t = SF2M.compute_lambda(t % 1)
            value_st = self.sf2m_score_model(t, xt)
            st = self.sf2m_score_model.compute_gradient(t, xt)
            positive_st = torch.relu(value_st)
            penalty = self.config['score_train']['lambda_penalty'] * torch.max(positive_st)
            
            score_loss = torch.mean((lambda_t[:, None] * st + eps) ** 2)
            if i % 100 == 0:
                self.logger.info(f"Max positive_st: {torch.max(positive_st)}")
                self.logger.info(f"Iteration {i}: score_loss = {score_loss.item():0.2f}")
            
            loss = score_loss + penalty
            loss.backward()
            sf2m_optimizer.step()
        
        torch.save(self.sf2m_score_model.state_dict(), os.path.join(self.exp_dir, 'score_model'))

    def final_training(self):
        """Phase 3: Final training phase"""
        self.logger.info('Final training phase')
        self.sf2m_score_model.load_state_dict(torch.load(os.path.join(self.exp_dir, 'score_model'), map_location=self.device))
        self.f_net.load_state_dict(torch.load(os.path.join(self.exp_dir, 'best_model'), map_location=self.device))
        
        optimizer = torch.optim.Adam(
            list(self.f_net.parameters()) + list(self.sf2m_score_model.parameters()),
            self.config['train']['lr']
        )
        
        if 'scheduler_step_size' in self.config['train']:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.config['train']['scheduler_step_size'],
                gamma=self.config['train']['scheduler_gamma']
            )
        else:
            scheduler = None
        
        dim = self.config['data']['dim']
        datatime0 = torch.zeros(self.df[self.df['samples']==0].x1.shape[0], dim)
        datatime0 = torch.tensor(self.df[self.df['samples']==0].iloc[:,1:dim+1].values, dtype=torch.float32).to(self.device)
        
        for param in self.f_net.parameters():
            param.requires_grad = True
        if self.config['use_pinn']:
            for param in self.sf2m_score_model.parameters():
                param.requires_grad = True
        else:
            for param in self.sf2m_score_model.parameters():
                param.requires_grad = False
        
        criterion = OT_loss1(which='emd', device=self.device)
        sample_size = (self.config['sample_size'],)
        
        l_loss, b_loss, g_loss = train_all(
            self.f_net, self.df, self.groups, optimizer, self.config['train']['epochs'],
            criterion=criterion,
            local_loss=True,
            global_loss=False,
            apply_losses_in_time=self.config['apply_losses_in_time'],
            hold_one_out=self.config['data']['hold_one_out'],
            hold_out=self.config['data']['hold_out'],
            sf2m_score_model=self.sf2m_score_model,
            hinge_value=self.config['hinge_value'],
            datatime0=datatime0,
            device=self.device,
            lambda_initial=self.config['train']['lambda_initial'],
            use_pinn=self.config['use_pinn'],
            use_penalty=self.config['use_penalty'],
            use_density_loss=self.config['use_density_loss'],
            lambda_density=self.config['lambda_density'],
            top_k=self.config['top_k'],
            sample_size=sample_size,
            relative_mass=self.relative_mass,
            initial_size=self.initial_size,
            sample_with_replacement=self.config['sample_with_replacement'],
            sigmaa=self.config['score_train']['sigma'],
            lambda_pinn=self.config['train']['lambda_pinn'],
            lambda_ot=self.config['train']['lambda_ot'],
            lambda_mass=self.config['train']['lambda_mass'],
            lambda_energy=self.config['train']['lambda_energy'],
            exp_dir=self.exp_dir,
            scheduler=scheduler,
            logger=self.logger
        )
        
        # Save final models
        if self.config['use_pinn']:
            torch.save(self.sf2m_score_model.state_dict(), os.path.join(self.exp_dir, 'score_model_final'))
        torch.save(self.f_net.state_dict(), os.path.join(self.exp_dir, 'model_final'))
        
        return l_loss, b_loss, g_loss

    def train(self):
        """Run complete training pipeline"""
        # Phase 1: Pretrain
        pretrain_losses = self.pretrain()

        # Phase 2: Train score model
        self.train_score_model()
        
        # Phase 3: Final training
        final_losses = self.final_training()
        
        return pretrain_losses, final_losses

def main():
    parser = argparse.ArgumentParser(description='Train DeepRUOT model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()
    
    # Load and merge configuration
    config = load_and_merge_config(args.config)
    
    # Create and run training pipeline
    pipeline = TrainingPipeline(config)
    pretrain_losses, final_losses = pipeline.train()
    
    return pretrain_losses, final_losses

if __name__ == '__main__':
    main() 