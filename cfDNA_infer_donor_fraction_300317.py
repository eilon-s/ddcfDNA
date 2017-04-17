#!/usr/bin/env python3

"""
Infers donor-cfDNA fraction from recipient genotype and cfDNA sequencing
Eilon Sharon 2017
"""

import os
import argparse
import gzip

import pandas as pd

import autograd.numpy as np # Thinly-wrapped version of Numpy
from autograd import grad
from scipy.optimize import minimize

import random


###############################################################################################################
# likelihhod helper functions
###############################################################################################################

def make_P_Ref_cond_source_mat():
    """
    2x9 matrix
    rows: source [recipient,donor]
    columns: recipient and donor genotypes 
        rec:   XX,XX,XX,XY,XY,XY,YY,YY,YY
        donor: XX,XY,YY,XX,XY,YY,XX,XY,YY
        where X is Reference allele and Y the alternative allele
    """
    pRefCondSource_mat = np.array(
        [ [1.0,1.0,1.0, 0.5,0.5,0.5, 0.0,0.0,0.0 ],
          [1.0,0.5,0.0, 1.0,0.5,0.0, 1.0,0.5,0.0 ]])
    
    return(pRefCondSource_mat)

def cal_recombination_prob(genetic_distances):
    """
    calculates the probability of an odd number of recombination events
    """
    theta_recomb_p = (1 - np.exp(-2*(genetic_distances/100)))/ 2
    return(theta_recomb_p)


def cal_ibd_y1_vec(genetic_distances,m):
    theta_recomb_p = cal_recombination_prob(genetic_distances)
    return( (1-theta_recomb_p)**(m-2) )

def cal_ibd_y2_vec(genetic_distances):
    theta_recomb_p = cal_recombination_prob(genetic_distances)
    return( (theta_recomb_p)**2 + (1-theta_recomb_p)**2 )


def make_sub_transition_matrix(genetic_distance,m,y1,y2):
    """
    transition matrix for one pair of recipient-donor chromosomes
    """
    sub_trans_mat = np.array( [[ 1 - ( (1-y1*y2) / (2**(m-1) - 1))  , (1-y1*y2) / (2**(m-1) - 1) ],
                               [ 1 - y1*y2 , y1*y2]])
    
    # TODO remove
    # in case the assumption that there aere at least two meiosis events does not hold
    # then it can only be twin or parent-child. in both cases there is no recombination and there is always IBD=1
    if (m < 2):
        if np.any(np.logical_or(sub_trans_mat<0,sub_trans_mat>1)):
            sub_trans_mat = np.array( [[0.0,1.0],
                                       [0.0,1.0]])

    return(sub_trans_mat)

def make_transition_matrix(genetic_distance,m_1,m_2,
                           m1_y1,m2_y1,
                           y2,
                           inf_genetic_distance = 1e6):
    
    if np.isfinite(genetic_distance) and  genetic_distance < inf_genetic_distance:
        
        sub_mat1 = make_sub_transition_matrix(genetic_distance,m_1,m1_y1,y2)
        sub_mat2 = make_sub_transition_matrix(genetic_distance,m_2,m2_y1,y2)
        
        # / 2 because intial state does not descriminate between which pair is in IBD in IBD 1
        #trans_mat = np.array(
        #                [[ sub_mat1[0,0]*sub_mat2[0,0], 
        #                   sub_mat1[0,0]*sub_mat2[0,1] + sub_mat1[0,1]*sub_mat2[0,0] , 
        #                   sub_mat1[0,1]*sub_mat2[0,1] ],
        #                 [ (sub_mat1[0,0]*sub_mat2[1,0] + sub_mat1[1,0]*sub_mat2[0,0])/2.0 ,  
        #                   (sub_mat1[0,0]*sub_mat2[1,1] + sub_mat1[0,1]*sub_mat2[1,0] + sub_mat1[1,0]*sub_mat2[0,1] + sub_mat1[1,1]*sub_mat2[0,0])/2.0,
        #                   (sub_mat1[0,1]*sub_mat2[1,1] + sub_mat1[1,1]*sub_mat2[0,1])/2.0],
        #                 [ sub_mat1[1,0]*sub_mat2[1,0], 
        #                   sub_mat1[1,0]*sub_mat2[1,1] + sub_mat1[1,1]*sub_mat2[1,0] , 
        #                   sub_mat1[1,1]*sub_mat2[1,1] ]])
        
        trans_mat = np.array(
                        [[ sub_mat1[0,0]*sub_mat2[0,0], 
                           sub_mat1[0,0]*sub_mat2[0,1],
                           sub_mat1[0,1]*sub_mat2[0,0] , 
                           sub_mat1[0,1]*sub_mat2[0,1] ],
                         [ sub_mat1[0,0]*sub_mat2[1,0],
                           sub_mat1[0,0]*sub_mat2[1,1],
                           sub_mat1[0,1]*sub_mat2[1,0],
                           sub_mat1[0,1]*sub_mat2[1,1] ],
                         [ sub_mat1[1,0]*sub_mat2[0,0],
                           sub_mat1[1,0]*sub_mat2[0,1],
                           sub_mat1[1,1]*sub_mat2[0,0],
                           sub_mat1[1,1]*sub_mat2[0,1] ],
                         [ sub_mat1[1,0]*sub_mat2[1,0], 
                           sub_mat1[1,0]*sub_mat2[1,1],
                           sub_mat1[1,1]*sub_mat2[1,0] , 
                           sub_mat1[1,1]*sub_mat2[1,1] ] ])
    else:
        marginal_p_noibd_1   = 1-2**(1-m_1)
        marginal_p_ibd_1     = 2**(1-m_1)
        marginal_p_noibd_2   = 1-2**(1-m_2)
        marginal_p_ibd_2     = 2**(1-m_2)
        
        trans_mat = np.array(
                        [[marginal_p_noibd_1*marginal_p_noibd_2 , marginal_p_noibd_1*marginal_p_ibd_2 , marginal_p_ibd_1*marginal_p_noibd_2, marginal_p_ibd_1*marginal_p_ibd_2],
                         [marginal_p_noibd_1*marginal_p_noibd_2 , marginal_p_noibd_1*marginal_p_ibd_2 , marginal_p_ibd_1*marginal_p_noibd_2, marginal_p_ibd_1*marginal_p_ibd_2],
                         [marginal_p_noibd_1*marginal_p_noibd_2 , marginal_p_noibd_1*marginal_p_ibd_2 , marginal_p_ibd_1*marginal_p_noibd_2, marginal_p_ibd_1*marginal_p_ibd_2],
                         [marginal_p_noibd_1*marginal_p_noibd_2 , marginal_p_noibd_1*marginal_p_ibd_2 , marginal_p_ibd_1*marginal_p_noibd_2, marginal_p_ibd_1*marginal_p_ibd_2]])
    
    return(trans_mat)


def make_P_measured_rec_geno_by_rec_geno_mat(geno_err):
    """
    9x3 matrix
    rows: recipeint true genotype: XX,XX,XX,XY,XY,XY,YY,YY,YY
    columns: recipient meausred genotype: XX,XY,YY
    """
    Eg = np.array(
        [ [(1-geno_err)**2, 2*geno_err*(1-geno_err) , geno_err**2],
          [(1-geno_err)**2, 2*geno_err*(1-geno_err) , geno_err**2],
          [(1-geno_err)**2, 2*geno_err*(1-geno_err) , geno_err**2],
          [geno_err*(1-geno_err), geno_err**2 + (1-geno_err)**2, geno_err*(1-geno_err)],
          [geno_err*(1-geno_err), geno_err**2 + (1-geno_err)**2, geno_err*(1-geno_err)],
          [geno_err*(1-geno_err), geno_err**2 + (1-geno_err)**2, geno_err*(1-geno_err)],
          [geno_err**2, 2*geno_err*(1-geno_err), (1-geno_err)**2],
          [geno_err**2, 2*geno_err*(1-geno_err), (1-geno_err)**2],
          [geno_err**2, 2*geno_err*(1-geno_err), (1-geno_err)**2] ])
    
    return(Eg)


def make_rec_measured_genotype_mat(block_data):
    """
    3xN matrix where N = number of SNPs (1 hot encoding)
    """
    # number of SNPs
    N = block_data.shape[0]
    
    recGeno = np.array( [(block_data.g1000_Allele1_receiver_p.values>0.75)*1.0,
                         np.logical_and(block_data.g1000_Allele1_receiver_p.values>0.25,block_data.g1000_Allele1_receiver_p.values<0.75) *1.0,
                         (block_data.g1000_Allele1_receiver_p.values<0.25)*1.0])
    
    return(recGeno)


def make_donor_geno_cond_recipient_geno(block_data, pop, minimal_allele_frequency, ibd = 0):
    """
    9xN matrix where N = number of SNPs
    
    rows: recipient and donor genotypes 
        rec:   XX,XX,XX,XY,XY,XY,YY,YY,YY
        donor: XX,XY,YY,XX,XY,YY,XX,XY,YY
    columns: SNPs in block
        
    """
    N = block_data.shape[0]
    gDonor_by_gRec_mat = np.zeros((9,N))
    
    pop_col = pop +'_AF'
    
    # elsewhere in the code alt is reffered as Y and ref is reffered as X
    
    # pop probabilities
    alt_pop_p = np.minimum(np.maximum(block_data[pop_col].values,minimal_allele_frequency), 1-minimal_allele_frequency)
    ref_pop_p = 1 - alt_pop_p
    
    #recipient probabilities
    alt_rec_p = block_data.g1000_Allele2_receiver_p.values
    ref_rec_p = block_data.g1000_Allele1_receiver_p.values

    
    if (ibd == 0):
        
        # rec XX, donor XX
        gDonor_by_gRec_mat[0,] = ref_pop_p**2
        # rec XX, donor XY
        gDonor_by_gRec_mat[1,] = 2*ref_pop_p*alt_pop_p
        # rec XX, donor YY
        gDonor_by_gRec_mat[2,] = alt_pop_p**2
        
        # rec XY, donor XX
        gDonor_by_gRec_mat[3,] = ref_pop_p**2
        # rec XY, donor XY
        gDonor_by_gRec_mat[4,] = 2*ref_pop_p*alt_pop_p
        # rec XY, donor YY
        gDonor_by_gRec_mat[5,] = alt_pop_p**2
        
        # rec YY, donor XX
        gDonor_by_gRec_mat[6,] = ref_pop_p**2
        # rec YY, donor XY
        gDonor_by_gRec_mat[7,] = 2*ref_pop_p*alt_pop_p
        # rec YY, donor YY
        gDonor_by_gRec_mat[8,] = alt_pop_p**2
        
    elif (ibd == 1):
        
        # rec XX, donor XX
        gDonor_by_gRec_mat[0,] = ref_pop_p
        # rec XX, donor XY
        gDonor_by_gRec_mat[1,] = alt_pop_p
        # rec XX, donor YY
        gDonor_by_gRec_mat[2,] = np.zeros_like(ref_pop_p)
        
        # rec XY, donor XX
        gDonor_by_gRec_mat[3,] = 0.5 * ref_pop_p
        # rec XY, donor XY
        gDonor_by_gRec_mat[4,] = (0.5 * ref_pop_p) + (0.5 * alt_pop_p)
        # rec XY, donor YY
        gDonor_by_gRec_mat[5,] = 0.5 * alt_pop_p
        
        # rec YY, donor XX
        gDonor_by_gRec_mat[6,] = np.zeros_like(ref_pop_p)
        # rec YY, donor XY
        gDonor_by_gRec_mat[7,] = ref_pop_p
        # rec YY, donor YY
        gDonor_by_gRec_mat[8,] = alt_pop_p
    
    elif (ibd == 2): #TO consider add remove epsilon
        
        # rec XX, donor XX
        gDonor_by_gRec_mat[0,] = np.ones_like(ref_pop_p)
        # rec XX, donor XY
        gDonor_by_gRec_mat[1,] = np.zeros_like(ref_pop_p)
        # rec XX, donor YY
        gDonor_by_gRec_mat[2,] = np.zeros_like(ref_pop_p)
        
        # rec XY, donor XX
        gDonor_by_gRec_mat[3,] = np.zeros_like(ref_pop_p)
        # rec XY, donor XY
        gDonor_by_gRec_mat[4,] = np.ones_like(ref_pop_p)
        # rec XY, donor YY
        gDonor_by_gRec_mat[5,] = np.zeros_like(ref_pop_p)
        
        # rec YY, donor XX
        gDonor_by_gRec_mat[6,] = np.zeros_like(ref_pop_p)
        # rec YY, donor XY
        gDonor_by_gRec_mat[7,] = np.zeros_like(ref_pop_p)
        # rec YY, donor YY
        gDonor_by_gRec_mat[8,] = np.ones_like(ref_pop_p)
    
    else:
        raise ValueError('Unknown IBD:' + str(ibd))
        
    return(gDonor_by_gRec_mat)  

 
def cal_P_ref_cfDNA(donorP_vec, pRefCondSource_mat, gDonor_by_gRec_ibd_mat, Eg_mat, recGeno_mat):
    return( np.dot(np.dot(donorP_vec,pRefCondSource_mat), 
                   gDonor_by_gRec_ibd_mat * np.dot(Eg_mat,recGeno_mat)) )


def cal_block_lemmision(donorP_vec, seq_err,
                       pRefCondSource_mat,
                       Eg_mat, recGeno_mat,
                       ref_cnts, alt_cnts,
                       ref_cnt_nonzero,alt_cnt_nonzero,
                       gDonor_by_gRec_ibd0_mat,gDonor_by_gRec_ibd1_mat,gDonor_by_gRec_ibd2_mat):
    
    
    P_ref_cfDNA_ibd0 =  cal_P_ref_cfDNA(donorP_vec, pRefCondSource_mat, gDonor_by_gRec_ibd0_mat, Eg_mat, recGeno_mat)
    P_ref_cfDNA_ibd1 =  cal_P_ref_cfDNA(donorP_vec, pRefCondSource_mat, gDonor_by_gRec_ibd1_mat, Eg_mat, recGeno_mat)
    P_ref_cfDNA_ibd2 =  cal_P_ref_cfDNA(donorP_vec, pRefCondSource_mat, gDonor_by_gRec_ibd2_mat, Eg_mat, recGeno_mat)
    
    lEmission_ibd0 = (np.sum(np.log( (   P_ref_cfDNA_ibd0[ref_cnt_nonzero]*(1-seq_err)) + ((1-P_ref_cfDNA_ibd0[ref_cnt_nonzero])*(seq_err)) ) * ref_cnts) +
                      np.sum(np.log( ((1-P_ref_cfDNA_ibd0[alt_cnt_nonzero])*(1-seq_err)) + (  P_ref_cfDNA_ibd0[alt_cnt_nonzero]*(seq_err)) ) * alt_cnts))
    
    lEmission_ibd1 = (np.sum(np.log( (   P_ref_cfDNA_ibd1[ref_cnt_nonzero]*(1-seq_err)) + ((1-P_ref_cfDNA_ibd1[ref_cnt_nonzero])*(seq_err)) ) * ref_cnts) +
                      np.sum(np.log( ((1-P_ref_cfDNA_ibd1[alt_cnt_nonzero])*(1-seq_err)) + (  P_ref_cfDNA_ibd1[alt_cnt_nonzero]*(seq_err)) ) * alt_cnts))
    
    
    lEmission_ibd2 = (np.sum(np.log( (   P_ref_cfDNA_ibd2[ref_cnt_nonzero]*(1-seq_err)) + ((1-P_ref_cfDNA_ibd2[ref_cnt_nonzero])*(seq_err)) ) * ref_cnts) +
                      np.sum(np.log( ((1-P_ref_cfDNA_ibd2[alt_cnt_nonzero])*(1-seq_err)) + (  P_ref_cfDNA_ibd2[alt_cnt_nonzero]*(seq_err)) ) * alt_cnts))

    lEmission = np.array([ lEmission_ibd0,lEmission_ibd1,lEmission_ibd1,lEmission_ibd2])
    
    return(lEmission)


###############################################################################################################
# likelihood function
###############################################################################################################

def negative_log_likelihood(x, obs):
    
    
    #print(x)
    
    p_donor      = x[0]
    seq_err      = x[1]
    geno_err     = x[2]
    p_ibd_haplo1 = x[3]
    p_ibd_haplo2 = x[4]
    
    m_1 = 1 - np.log2(p_ibd_haplo1)
    m_2 = 1 - np.log2(p_ibd_haplo2)
    
    
    #print('m_1:')
    #print(p_ibd_haplo1)
    #print(m_1)
    #print(p_ibd_haplo1)
    
    if (m_1 < 1):
      print('m_1 samller than one')
      print(m_1)
      print(p_ibd_haplo1)
    
    if (m_2 < 1):
      print('m_2 samller than one')
      print(m_2)
      print(p_ibd_haplo1)
    
    assert(m_1 >= 1.0)
    assert(m_2 >= 1.0)
    
    ###############################################################################################################
    # pre-calculating matrices that depend on other parameters (each evaluation)
    ###############################################################################################################

    # for the transition matrices
    
    inf_genetic_distance = obs['inf_genetic_distance']
    genetic_distances = obs['genetic_distances']
    y2_vec = obs['y2_vec']
    m1_y1_vec = cal_ibd_y1_vec(genetic_distances,m_1)
    m2_y1_vec = cal_ibd_y1_vec(genetic_distances,m_2)
    
    #m1_y2_vec = cal_ibd_y2_vec(genetic_distances) # cal_ibd_y2_vec(genetic_distances,m_1) # TODO move to precalculations
    #m2_y2_vec = m1_y2_vec # cal_ibd_y2_vec(genetic_distances,m_2)
    
    
    #transition_mats = make_transition_matrices(obs['genetic_distances'], p_ibd_haplo1, p_ibd_haplo2, inf_genetic_distance = obs['inf_genetic_distance'])

    # donor fraction vector
    donorP_vec = np.array([1-p_donor,p_donor])

    # genotyping error matrix
    Eg_mat = make_P_measured_rec_geno_by_rec_geno_mat(geno_err)


    #lemission_vecs = [None] * obs['num_blocks']
    
    ll_of_stats = np.zeros(4)
    
    i=0
    for block_i, block_data in obs['blocks_data']:
        #if (i % 300 == 0):
        #    print("calculating emmision block:", block_i)
        
        
        ltrans_mat = np.log(make_transition_matrix(genetic_distances[i],m_1,m_2,
                                           m1_y1_vec[i],m2_y1_vec[i],
                                           y2_vec[i],
                                           inf_genetic_distance = inf_genetic_distance))
        
        lemiss_vec = cal_block_lemmision(donorP_vec, seq_err,
                                         obs['pRefCondSource_mat'],
                                         Eg_mat, obs['recGeno_mats'][i],
                                         obs['ref_cnt_vecs'][i], obs['alt_cnt_vecs'][i],
                                         obs['ref_cnt_nonzero_vecs'][i],obs['alt_cnt_nonzero_vecs'][i],
                                         obs['gDonor_by_gRec_ibd0_mats'][i],
                                         obs['gDonor_by_gRec_ibd1_mats'][i],
                                         obs['gDonor_by_gRec_ibd2_mats'][i])
        
        in_ll = np.transpose(np.array([ll_of_stats])) + ltrans_mat
        #ll_of_stats = np.logaddexp(np.logaddexp(in_ll[0,],in_ll[1,]),in_ll[2,]) +  lemiss_vec
        ll_of_stats = np.max(in_ll,axis=0) +  lemiss_vec


        i=i+1
        
    #nll =  -np.logaddexp(np.logaddexp(ll_of_stats[0],ll_of_stats[1]),ll_of_stats[2])
    nll =  -np.max(ll_of_stats)
    
    
    #print("nll")
    #print(nll)
    
    return(nll)

# nll gradient
grad_negative_log_likelihood = grad(negative_log_likelihood)

###############################################################################################################
# preparing the observed data and pre-calculations
###############################################################################################################

def make_obs_from_data(data_df,verbose = True):
    obs = {}
    obs['inf_genetic_distance'] = 1e6
    obs['genetic_blocks'] = data_df.genetic_blocks.unique()
    obs['num_blocks'] = obs['genetic_blocks'].shape[0]
    #obs['genetic_distances'] = data_df['genetic_avg_distance_FromPrevBlock'].groupby(data_df['genetic_blocks']).mean().values
    tmp_genetic_distances = data_df['genetic_avg_distance_FromPrevBlock'].groupby(data_df['genetic_blocks']).mean().values
    tmp_genetic_distances[np.isinf(tmp_genetic_distances)] = obs['inf_genetic_distance']
    obs['genetic_distances'] = tmp_genetic_distances
    obs['blocks_data'] = data_df.groupby('genetic_blocks')
    
    obs['y2_vec']  = cal_ibd_y2_vec(obs['genetic_distances'])
    
    ###########################
    # pre-calculating matrices
    ###########################
    
    obs['pRefCondSource_mat'] = make_P_Ref_cond_source_mat()
    
    # measured recipient genotype
    recGeno_mats = [None] * obs['num_blocks']
    ref_cnt_vecs = [None] * obs['num_blocks']
    alt_cnt_vecs = [None] * obs['num_blocks']
    
    ref_cnt_nonzero_vecs = [None] * obs['num_blocks']
    alt_cnt_nonzero_vecs = [None] * obs['num_blocks']
    
    i=0
    for block_i, block_data in obs['blocks_data']:
        if (verbose and i % 300 == 0):
            print("pre-calculating block: %d" % (block_i))
        
        recGeno_mats[i] = make_rec_measured_genotype_mat(block_data)
        
        cur_ref_cnt_vec = block_data['g1000_Allele1_seq_cnt'].values
        cur_alt_cnt_vec = block_data['g1000_Allele2_seq_cnt'].values
        
        ref_cnt_nonzero_vecs[i] = cur_ref_cnt_vec > 0.9
        alt_cnt_nonzero_vecs[i] = cur_alt_cnt_vec > 0.9
        
        ref_cnt_vecs[i] = cur_ref_cnt_vec[ref_cnt_nonzero_vecs[i]]
        alt_cnt_vecs[i] = cur_alt_cnt_vec[alt_cnt_nonzero_vecs[i]]
        
        i=i+1
    
    obs['recGeno_mats'] = recGeno_mats
    obs['ref_cnt_vecs'] = ref_cnt_vecs
    obs['alt_cnt_vecs'] = alt_cnt_vecs
    obs['ref_cnt_nonzero_vecs'] = ref_cnt_nonzero_vecs
    obs['alt_cnt_nonzero_vecs'] = alt_cnt_nonzero_vecs
    
    
    obs['number_of_SNPs'] = data_df.shape[0]
    obs['number_of_reads_mapped_to_SNPs'] = data_df.g1000_Allele1_seq_cnt.sum() + data_df.g1000_Allele2_seq_cnt.sum()
    obs['number_of_genetic_blocks'] = data_df.genetic_blocks.unique().shape[0]
    
    print("Number of SNPs: %d" % (obs['number_of_SNPs']))
    print("Number of reads mapped to SNPs: %d" % (obs['number_of_reads_mapped_to_SNPs']))
    print("Number of genetic blocks: %d" % (obs['number_of_genetic_blocks']))

    return(obs)


###############################################################################################################
# optimization
###############################################################################################################


def update_obs_with_pop_gDonor_by_gRec(obs,pop,minimal_allele_frequency,verbose):
  
    """
    pre-calculating matrices that depend on the population (once)
    """
    
    # P(donor genotype | recipeint genotype, pop, ibd)
    gDonor_by_gRec_ibd0_mats = [None] * obs['num_blocks']
    gDonor_by_gRec_ibd1_mats = [None] * obs['num_blocks']
    gDonor_by_gRec_ibd2_mats = [None] * obs['num_blocks']
    
    i=0
    for block_i, block_data in obs['blocks_data']:
        if (verbose and i % 300 == 0):
            print("pre-calculating (conditional on population) block: %d" % (block_i))
        
        gDonor_by_gRec_ibd0_mats[i] = make_donor_geno_cond_recipient_geno(block_data, pop, 
                                                                          minimal_allele_frequency, ibd = 0)
        gDonor_by_gRec_ibd1_mats[i] = make_donor_geno_cond_recipient_geno(block_data, pop, 
                                                                          minimal_allele_frequency, ibd = 1)
        gDonor_by_gRec_ibd2_mats[i] = make_donor_geno_cond_recipient_geno(block_data, pop, 
                                                                          minimal_allele_frequency, ibd = 2)
        
        i=i+1
    
    obs['gDonor_by_gRec_ibd0_mats'] = gDonor_by_gRec_ibd0_mats
    obs['gDonor_by_gRec_ibd1_mats'] = gDonor_by_gRec_ibd1_mats
    obs['gDonor_by_gRec_ibd2_mats'] = gDonor_by_gRec_ibd2_mats
    
    return(obs)



def optimize_for_given_pop(obs,pop,x0,x_bounds,minimal_allele_frequency,verbose = True):
    """
    notice it updates gDonor_by_gRec_ibd<0|1|2>_mats for the current population
    """
    
    obs = update_obs_with_pop_gDonor_by_gRec(obs,pop,minimal_allele_frequency,verbose)
    
    if verbose:
      opt_options = {'disp': True}
    else:
      opt_options = {'disp': False}
    
    
    opt_val = minimize(negative_log_likelihood, x0, args=(obs,), method='L-BFGS-B', 
                   jac=grad_negative_log_likelihood, 
                   bounds=x_bounds, options = opt_options)
    
    print('-' * 20)
    print('Population: %s' % (pop))
    print('Donor fraction: %f' % (opt_val.x[0]))
    print('Sequencing error: %f' % (opt_val.x[1]))
    print('Genotyping error: %f' % (opt_val.x[2]))
    print('P(IBD) first pair: %f' % (opt_val.x[3]))
    print('P(IBD) second pair: %f' %(opt_val.x[4]))
    print('nll: %f' % (opt_val.fun))
    print('-' * 20)
    
    return(opt_val)


###############################################################################################################
# main function
###############################################################################################################

def main():

    __version__ = "0.32"
  
    parser = argparse.ArgumentParser(description="Infers donor-cfDNA fraction from recipient genotype and cfDNA sequencing", prog="cfDNA1G")

    # input files

    parser.add_argument("input_data_filename", type=str, help= "tab-separated values (tsv) file produced by the cfDNA1G pipeline. Each SNP is a raw.")

    #output files
    
    parser.add_argument("output_filename", type=str, 
                      help="tab-separated values (tsv) file. a table describing the result for each tested population. the selected population is indicated")
    
    
    # options

    parser.add_argument("-i", "--infer_relatedness", dest='infer_relatedness', type=str, default='LearnAll',
                      help='whether to infer relatednes <LearnAll(default)|LearnUnknown|ByInput>')
    
    parser.add_argument("-r", "--input_relatedness", dest='input_relatedness', type=str, default='UNKNOWN',
                      help='known recipient-donor relatedness <UNKNOWN(default)|SIBLINGS|NOT_RELATED>')
    
    parser.add_argument("-n", "--num_optimization_starting_points", dest='num_optimization_starting_points', default=4, type=int,
                      help=('Number of starting points for the optimization algorithm. Positive integer >=3, defualt is 4. First two starting points are similar for all populations, the rest are uniformly distributed within bounds'))
    

    parser.add_argument("-m", "--minimal_allele_frequency", dest='minimal_allele_frequency', default=1e-5, type=float,
                      help=('minimal allele frequency in the population. Zero probabilities will be tranfrom to this value'))
    
    parser.add_argument("-s", "--sample_id", dest='sample_id', default="SampleID", type=str,
                      help=('unique sample ID'))
    
    parser.add_argument("-p", "--run_only_these_pops", dest='run_only_these_pops', default=[], nargs='*',
                      help=('a list of populations. if provided will test only these populations. The default is an emtpy list, in this case all populations will be tested'))
    
    parser.add_argument("-v", "--verbose", dest='verbose',  action='store_true',
                        help=('print optimization progress'))
    
    parser.add_argument('--version', action='version', version='%(prog)s {version}'.format(version=__version__))

    
    
    args = parser.parse_args()

    
    assert(args.num_optimization_starting_points >= 3)
    assert(args.minimal_allele_frequency >= 0.0 and args.minimal_allele_frequency <= 1.0)


    print("Running version %s" %(__version__))

    # running the function
    
    print('Loading file: %s' % (args.input_data_filename))
    
    data_df = pd.read_table(args.input_data_filename, '\t')
    
    # pre calculations
    print('run precalculations that are similar for all populations')
    
    obs = make_obs_from_data(data_df, args.verbose)
    
    # setting X0 and bounds
    # bounds are dependent on infer_relatedness and input_relatedness
    
    infer_relatedness = args.infer_relatedness
    input_relatedness = args.input_relatedness
    
    print("Input infer_relatedness: %s" % (infer_relatedness))
    print("Input input_relatedness: %s" % (input_relatedness))
    
    
    if (input_relatedness in ["siblings", "Siblings", "SIBLINGS", "sibling", "Sibling", "SIBLING"]):
        input_relatedness = "SIBLINGS"
    elif (input_relatedness in ["UnrelatedMatched"]):
        input_relatedness = "NOT_RELATED"
    else: # ,"Haplocompatable", NA,"NA","SingleCord",
        input_relatedness = "UNKNOWN"

    if (infer_relatedness == "ByInput" ):
        input_relatedness = input_relatedness
    elif (infer_relatedness == "LearnAll"):
        input_relatedness = "LEARN"
    elif (infer_relatedness == "LearnUnknown"):
        if (input_relatedness == "UNKNOWN"):
            input_relatedness = "LEARN"
        else:
            input_relatedness = input_relatedness
    elif (infer_relatedness == "LearnUnknownAndNotRelated" ):
        if(input_relatedness in ["UNKNOWN","NOT_RELATED"]):
            input_relatedness = "LEARN"
        else:
            input_relatedness = input_relatedness
    elif (infer_relatedness == "TreatAsUnrelated" ):
        input_relatedness = "NOT_RELATED"
    else:
        raise ValueError("ERROR unknown infer_relatedness:%s" % (infer_relatedness))
    
    
    print("Run assuming relatedness: %s" % (input_relatedness))
    
    eps = 1e-9
    if (input_relatedness in ["LEARN","UNKNOWN"]):
      # default
      x_bounds = [(eps,1-eps), (eps, 1e-2), (eps, 1e-3),(eps, 0.5),(eps, 0.5) ]
    elif (input_relatedness == "SIBLINGS"):
      x_bounds = [(eps,1-eps), (eps, 1e-2), (eps, 1e-3),(0.5, 0.5),(0.5, 0.5) ]
    elif (input_relatedness == "NOT_RELATED"):
      x_bounds = [(eps,1-eps), (eps, 1e-2), (eps, 1e-3),(eps, eps),(eps, eps)]
    
    
    #rndstate = random.getstate()
    #random.setstate(rndstate)
    
    x0 = np.zeros(5)
    for i in range(x0.shape[0]):
      x0[i] = random.uniform(x_bounds[i][0],x_bounds[i][1])
    
    x0_joint_for_pops_rand = x0.copy()

    x0 = np.zeros(5)
    # for the first iteration the relatedness starts from equl and reasonable starting point
    x0[0] = 0.1
    x0[1] = 1e-4
    x0[2] = 1e-5
  
    if (input_relatedness in ["LEARN","UNKNOWN"]):
      x0[3] = 0.01
      x0[4] = 0.01
    elif (input_relatedness == "SIBLINGS"):
      x0[3] = 0.5
      x0[4] = 0.5
    
    x0_joint_for_pops = x0.copy()
    #x0 = [0.1,1e-4,1e-5,1e-5,1e-5]
    
    
    # iterating over populations
    
    pops = [col.split('_')[0] for col in data_df.columns if '_AF' in col]
    
    # running 
    if args.run_only_these_pops:
      print('Filtering populations according to input')
      pops = [pop for pop in pops if pop in args.run_only_these_pops]
    
    
    print('Found %d populations (either all populations in the file or those that intersect --run_only_these_pops argument):' % (len(pops)))
    for p, pop in enumerate(pops):
        print("%d. %s" % (p+1,pop))
    
    pops_opt = [None] * len(pops)
    
    for p, pop in enumerate(pops):
      
        print('Infering donor cfDNA level assuming donor population is: %s (%d out of %d)' % (pop,p+1,len(pops)))
      
        for s in range(args.num_optimization_starting_points):
          
            if (s == 0) :
              x0 = x0_joint_for_pops.copy()
            elif (s == 1) :
              x0 = x0_joint_for_pops_rand.copy()
            else:
              for x_i in range(x0.shape[0]):
                  x0[x_i] = random.uniform(x_bounds[x_i][0],x_bounds[x_i][1])
              
            cur_opt = optimize_for_given_pop(obs,pop,x0,x_bounds, args.minimal_allele_frequency, args.verbose)
            
            if (s == 0 or cur_opt.fun < best_pop_opt.fun or np.isnan(best_pop_opt.fun)):
                best_pop_opt = cur_opt
        
        pops_opt[p] = best_pop_opt
    
    
    print("Preparing output table")
    out_dict = {'Sample' : [args.sample_id] * len(pops),
                'Population': pops,
                'IsDonorPop': [0] * len(pops),
                'P_donor' : [opt_val.x[0] for opt_val in pops_opt],
                'P_recipient' : [1-opt_val.x[0] for opt_val in pops_opt],
                'Sequencing_error' : [opt_val.x[1] for opt_val in pops_opt],
                'Genotyping_error' : [opt_val.x[2] for opt_val in pops_opt],
                'P_IBD_pair1' : [opt_val.x[3] for opt_val in pops_opt],
                'P_IBD_pair2' : [opt_val.x[4] for opt_val in pops_opt],
                'nll' : [opt_val.fun for opt_val in pops_opt],
                'number_of_SNPs' : [obs['number_of_SNPs']] * len(pops),
                'number_of_reads_mapped_to_SNPs' : [obs['number_of_reads_mapped_to_SNPs']] * len(pops),
                'number_of_genetic_blocks' : [obs['number_of_genetic_blocks']] * len(pops),
                'input_relatedness' :  [input_relatedness] * len(pops),
                'num_optimization_starting_points' :  [args.num_optimization_starting_points] * len(pops),
                'minimal_allele_frequency' : [args.minimal_allele_frequency] * len(pops) }
    
    #out_df = pd.DataFrame(out_dict)

    out_df = pd.DataFrame({'Sample' : out_dict['Sample']})
    out_df['Population'] = out_dict['Population']
    out_df['IsDonorPop'] = out_dict['IsDonorPop']
    out_df['P_donor'] = out_dict['P_donor']
    out_df['P_recipient'] = out_dict['P_recipient']
    out_df['Sequencing_error'] = out_dict['Sequencing_error']
    out_df['Genotyping_error'] = out_dict['Genotyping_error']
    out_df['P_IBD_pair1'] = out_dict['P_IBD_pair1']
    out_df['P_IBD_pair2'] = out_dict['P_IBD_pair2']
    out_df['nll'] = out_dict['nll']
    out_df['number_of_SNPs'] = out_dict['number_of_SNPs']
    out_df['number_of_reads_mapped_to_SNPs'] = out_dict['number_of_reads_mapped_to_SNPs']
    out_df['number_of_genetic_blocks'] = out_dict['number_of_genetic_blocks']
    out_df['input_relatedness'] = out_dict['input_relatedness']
    out_df['num_optimization_starting_points'] = out_dict['num_optimization_starting_points']
    out_df['minimal_allele_frequency'] = out_dict['minimal_allele_frequency']
    
    print("Selecting donor population")
    out_df.ix[out_df['nll'].idxmin(),'IsDonorPop'] = 1
    
    print('='*60)
    print("Results for sample: %s" % (out_df.Sample[out_df.IsDonorPop == 1].values[0]))
    print("Selected donor population: %s" % (out_df.Population[out_df.IsDonorPop == 1].values[0]))
    print("Donor cfDNA fraction: %f" % out_df.P_donor[out_df.IsDonorPop == 1].values[0])
    print("Recipient cfDNA fraction: %f" % out_df.P_recipient[out_df.IsDonorPop == 1].values[0])
    print("Sequencing error: %f" % out_df.Sequencing_error[out_df.IsDonorPop == 1].values[0])
    print("Genotyping error: %f" % out_df.Genotyping_error[out_df.IsDonorPop == 1].values[0])
    print("IBD pair 1 of recipient-donor chromosomes: %f" % out_df.P_IBD_pair1[out_df.IsDonorPop == 1].values[0])
    print("IBD pair 2 of recipient-donor chromosomes: %f" % out_df.P_IBD_pair2[out_df.IsDonorPop == 1].values[0])
    print("Negative log-likelihood: %f" % out_df.nll[out_df.IsDonorPop == 1].values[0])
    print('='*60)
    
    print("Writing output file: %s" % (args.output_filename))
    out_df.to_csv(args.output_filename, sep='\t', index = False, header = True) # compression='gzip'


    
if __name__ == '__main__':
  main()



