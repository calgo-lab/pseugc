from __future__ import annotations
from datasets import Dataset, DatasetDict
from io import StringIO
from IPython.display import display
from matplotlib.gridspec import GridSpec
from pandas import DataFrame
from pandas.core.indexes.numeric import Int64Index
from pathlib import Path
from shutil import copy, rmtree
from sklearn.model_selection import KFold
from somajo import SoMaJo
from tqdm import tqdm
from typing import Tuple, List, Set, Dict, Union, Any
from yaml.loader import SafeLoader

import glob
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import re
import torch
import yaml



class CodealltagDataProcessor:
    
    # constructor
    def __init__(self, data_version: str, config_path: List[str]):
        self.__data_version = data_version
        self.__config_path_str = os.path.join(*config_path)
        self.__config = self.__load_config()
    
    # public
    
    def get_data_version(self) -> str:
        return self.__data_version

    def get_config_path_str(self) -> str:
        return self.__config_path_str

    def print_config(self) -> None:
        self.__read_file(self.config_path_str, show=True)

    def reload_config(self) -> None:
        self.__config = self.__load_config()

    def get_random_seed(self) -> int:
        return self.__config[self.__get_config_key_random_seed()]

    def get_data_dir_path(self) -> str:
        return os.path.join(*self.__config[self.__get_config_key_data_dir_path()])
    
    def get_max_file_size(self) -> int:
        return self.__config[self.__get_config_key_max_file_size()]

    def get_xl_dir_path(self) -> str:
        return os.path.join(self.data_dir_path, self.__get_xl_dir_name())

    def get_xl_ann_dir_path(self) -> str:
        return os.path.join(self.data_dir_path, self.__get_xl_ann_dir_name())

    # processing

    def check_for_missing_annotations_files(self) -> List[str]:
        missing_annotations = []
        self.__check_and_print_status_for_root_dir()
        self.__check_and_print_status_for_content_in_root_dir(missing_annotations)
        return missing_annotations

    def get_category_path_df(self, force_create: bool = False) -> DataFrame:
        if not os.path.exists(self.__get_category_path_df_file_path()) or force_create:
            self.__create_category_path_df()
        return pd.read_csv(self.__get_category_path_df_file_path(), index_col=0)
    
    def get_absolute_email_file_path(self, file_path: str) -> str:
        return os.path.join(
            self.xl_dir_path, file_path
        ).replace(
            self.__get_file_extension_ann(), self.__get_file_extension_txt()
        )

    def read_email(self, file_path: str, show: bool = False) -> Tuple[str, str]:
        file_path = self.get_absolute_email_file_path(file_path)
        return file_path, self.__read_file(file_path, show)

    def get_absolute_annotations_file_path(self, file_path: str) -> str:
        return os.path.join(
            self.xl_ann_dir_path, file_path
        ).replace(
            self.__get_file_extension_txt(), self.__get_file_extension_ann()
        )
     
    def read_annotations(self, file_path: str, show: bool = False) -> Tuple[str, str]:
        file_path = self.get_absolute_annotations_file_path(file_path)
        return file_path, self.__read_file(file_path, show)

    def get_category_from_file_path(self, file_path: str) -> str:
        splits = file_path.split(os.path.sep)
        pattern = re.compile('^' + self.__get_xl_dir_prefix() + '[A-Z]+')
        for index, split in enumerate(splits):
            if pattern.match(split):
                return re.sub(self.__get_xl_dir_prefix(), '', splits[index])
        return file_path

    def get_file_size_nth_percentile(self, percentile: float):
        return np.quantile(self.get_category_path_df().FileSize, percentile)

    def get_annotation_file_path_from_email_file_path(self, file_path: str):
        return file_path.replace(
            (os.path.sep + self.__get_xl_dir_name() + os.path.sep),
            (os.path.sep + self.__get_xl_ann_dir_name() + os.path.sep)
        ).replace(
            self.__get_file_extension_txt(),
            self.__get_file_extension_ann()
        )

    def get_annotation_tuples_by_file(self, file_path: str, show: bool = False) -> List[Tuple]:
        file_path_annotations, content_annotations = self.read_annotations(file_path)
        annotation_tuples = list()
        if content_annotations is not None:
            lines = content_annotations.splitlines()
            for item in lines:
                attributes = re.split(self.__get_whitespace_regex(), item)
                annotation_tuples.append((
                    file_path_annotations.replace(self.xl_ann_dir_path + os.sep, ''),
                    attributes[0],
                    attributes[1],
                    attributes[2],
                    attributes[3],
                    attributes[4] if len(attributes) == 4 else " ".join(attributes[4:len(attributes)])
                ))
        if show:
            print(annotation_tuples)
            print()

        return annotation_tuples

    def get_annotation_df_by_file(self, file_path: str, show: bool = False) -> DataFrame:
        annotation_df_by_file = pd.DataFrame(
            self.get_annotation_tuples_by_file(file_path),
            columns=self.__get_annotation_df_columns()
        )
        if show:
            display(annotation_df_by_file)
            print()

        return annotation_df_by_file

    def get_annotation_df(self, use_only_paths: List[str] = None, force_create: bool = False) -> DataFrame:
        if not os.path.exists(self.__get_annotation_df_file_path()) or force_create:
            self.__create_annotation_df(use_only_paths)
        return pd.read_csv(self.__get_annotation_df_file_path(), index_col=0)
    
    def get_same_entity_labels_path_df(self, reference_cdp: CodealltagDataProcessor, force_create: bool = False) -> DataFrame:
        same_entity_labels_path_df_path = self.__get_same_entity_labels_path_df_path(reference_cdp)
        if not os.path.exists(same_entity_labels_path_df_path) or force_create:
            self.__create_same_entity_labels_path_df(reference_cdp)
        return pd.read_csv(same_entity_labels_path_df_path, index_col=0)
    
    def get_same_entity_labels_path_df_for_sample_size(self, 
                                                       reference_cdp: CodealltagDataProcessor,
                                                       sample_size: int,
                                                       force_create: bool = False) -> DataFrame:
        
        same_entity_labels_path_df_path_for_sample_size = self.__get_same_entity_labels_path_df_path_for_sample_size(
            reference_cdp,
            sample_size
        )
        if not os.path.exists(same_entity_labels_path_df_path_for_sample_size) or force_create:
            self.__prepare_3K_10K_samples(reference_cdp)
        return pd.read_csv(same_entity_labels_path_df_path_for_sample_size, index_col=0)
    
    def tokenize_with_somajo(self, text: str, label: Union[str, None] = None) -> List[str]:
        tokenizer = self.__get_somajo_tokenizer()
        tokenized_texts: List[str] = list()
        for sentence in tokenizer.tokenize_text([text]):
            for token in sentence:
                tokenized_texts.append(token.text)

        if label and len(tokenized_texts) > 0:
            labeled_texts: List[str] = list()
            if label == "O":
                for item in tokenized_texts:
                    labeled_texts.append(item + " " + label)
            else:
                labeled_texts.append(tokenized_texts[0] + " " + "B-" + label)
                for item in tokenized_texts[1:len(tokenized_texts)]:
                    labeled_texts.append(item + " " + "I-" + label)
            return labeled_texts

        return tokenized_texts
    
    def tokenize_with_somajo_and_annotation_df(self, email_text: str, annotation_df: DataFrame) -> str:
        
        non_entity_texts: List[str] = list()
        entity_texts: List[str] = list()

        text_stretch: str = ""
        tokenized_list: List[str] = list()
        tokenized_text: str = ""
        next_start: int = 0
        for index, row in annotation_df.iterrows():
            if int(row.Start) > next_start:
                text_stretch = email_text[next_start:int(row.Start)]
                non_entity_texts.append(text_stretch)
                tokenized_list = self.tokenize_with_somajo(text_stretch, "O")
                if len(tokenized_list) > 0:
                    tokenized_text += "\n".join(tokenized_list) + "\n"

            text_stretch = email_text[int(row.Start): int(row.End)]
            entity_texts.append(text_stretch)
            tokenized_list = self.tokenize_with_somajo(text_stretch, row.Label)
            tokenized_text += "\n".join(tokenized_list) + "\n"
            next_start = int(row.End)

        if len(email_text) > next_start:
            text_stretch = email_text[next_start: len(email_text)]
            non_entity_texts.append(text_stretch)
            tokenized_list = self.tokenize_with_somajo(text_stretch, "O")
            if len(tokenized_list) > 0:
                tokenized_text += "\n".join(tokenized_list) + "\n"

        return tokenized_text
    
    def get_output_sequence_for_seq2seq_ner_model(self, email_file_path: str) -> str:
        annotation_df = self.get_annotation_df_by_file(email_file_path)
        return (annotation_df.Label + ": " + annotation_df.Token).str.cat(sep="; ")
    
    def get_output_sequence_for_seq2seq_ner_pg_model(self,
                                                     email_file_path: str, 
                                                     reference_cdp: CodealltagDataProcessor) -> str:
    
        annotation_df = self.get_annotation_df_by_file(email_file_path)
        annotation_df_reference = reference_cdp.get_annotation_df_by_file(email_file_path)
        return (
            annotation_df.Label +
            ": " +
            annotation_df.Token + 
            " **" + 
            annotation_df_reference.Token +
            "**"
        ).str.cat(sep="; ")
    
    def get_model_data_for_sample_size(self, 
                                       reference_cdp: CodealltagDataProcessor,
                                       sample_size: int,
                                       force_create: bool = False) -> DataFrame:
        
        selpdf = self.get_same_entity_labels_path_df_for_sample_size(reference_cdp, sample_size)
        columns = selpdf.columns.values.tolist()
        if ('InputType1' not in columns or
            'InputType2' not in columns or
            'OutputType1' not in columns or
            'OutputType2' not in columns or 
            force_create):
            return self.__prepare_model_data_for_sample_size(reference_cdp, sample_size)
        else:
            return selpdf
    
    def get_train_dev_test_ids_for_sample_size(self, 
                                               reference_cdp: CodealltagDataProcessor,
                                               sample_size: int,
                                               test_data_percentage: float = 0.20,
                                               n_fold: int = 5) -> List[Tuple]:
        fold_tuples = list()
        sample_df = self.get_model_data_for_sample_size(reference_cdp, sample_size)
        sample_df.reset_index(drop=True, inplace=True)
        sample_df = sample_df.sample(frac=1, random_state = sample_size, ignore_index=True)
        
        random.seed(sample_size)
        test_indices = random.sample(sample_df.ID.tolist(), int(sample_size * test_data_percentage))
        
        sample_df = sample_df[~sample_df.ID.isin(test_indices)]
        sample_df.reset_index(drop=True, inplace=True)
        
        splits = list(KFold(n_splits=n_fold, shuffle=True, random_state=sample_size).split(sample_df.index.to_numpy()))
        train_dev_k_folds = self.get_train_dev_folds(n_fold)
        for index, fold in enumerate(train_dev_k_folds):
            train_indices = list()
            fold_train_indices = fold[1]
            for fold_train_index in fold_train_indices:
                train_indices += list(splits[fold_train_index][1])
            dev_indices = list(splits[fold[2][0]][1])
            fold_tuples.append((
                index + 1,
                sample_df[sample_df.index.isin(train_indices)].ID.tolist(),
                sample_df[sample_df.index.isin(dev_indices)].ID.tolist(),
                test_indices
            ))
        return fold_tuples
    
    def get_train_dev_test_datasetdict_for_sample_size(self,
                                                       reference_cdp: CodealltagDataProcessor,
                                                       sample_size: int,
                                                       k: int = 1,
                                                       test_data_percentage: float = 0.20,
                                                       n_fold: int = 5) -> DatasetDict:
        
        sample_df = self.get_model_data_for_sample_size(reference_cdp, sample_size)
        sample_df.reset_index(drop=True, inplace=True)
        
        fold_tuples = self.get_train_dev_test_ids_for_sample_size(reference_cdp,
                                                                  sample_size,
                                                                  test_data_percentage,
                                                                  n_fold)
        
        kth_tuple = fold_tuples[k-1]
                
        train_ds = Dataset.from_pandas(sample_df[sample_df.ID.isin(kth_tuple[1])])
        dev_ds = Dataset.from_pandas(sample_df[sample_df.ID.isin(kth_tuple[2])])
        test_ds = Dataset.from_pandas(sample_df[sample_df.ID.isin(kth_tuple[3])])

        return DatasetDict({
            'train': train_ds,
            'dev': dev_ds,
            'test': test_ds
        })
    
    # plot-related-functions
    def plot_category_wise_frequency(self, input_dataframe: DataFrame, export: bool = False, ext: str = 'png') -> None:
    
        #========================================================
        category_wise_counts = input_dataframe.groupby(
            'Category'
        ).agg(
            Count=pd.NamedAgg(column='FilePath', aggfunc='count')
        ).sort_values(
            by='Category', ascending=True
        )
        #========================================================

        #==========================================================
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
        fig.subplots_adjust(wspace=0.05)
        #==========================================================

        #=================================================================================================
        max_value = category_wise_counts.Count.max()
        buffer = max_value * 0.25
        rightmost_tick = max_value + buffer
        first_tick = max_value * 0.01
        ticks = np.linspace(first_tick, rightmost_tick, 5)
        
        category_wise_counts.Count.plot.barh(
            ax=axes[0],
            color='#4DD0E1',
            alpha=0.8
        )
        
        axes[0].set_xlim(0, rightmost_tick)
        axes[0].set_xticks(
            ticks,
            [self.__human_format_xaxis(tick) for tick in ticks],
            fontsize=15
        )
        axes[0].tick_params(axis='y', labelrotation=360, labelsize=15)
        axes[0].grid(visible=False, axis='y')
        axes[0].xaxis.grid(True, linestyle='--', alpha=0.7)
        axes[0].bar_label(
            axes[0].containers[0],
            labels=[self.__human_format_for_bars(value) for value in category_wise_counts.Count.tolist()],
            padding=2,
            color='r',
            fontsize=13
        )
        axes[0].set_xlabel('Email Count', labelpad=15, fontsize=15)
        axes[0].set_ylabel('Category', labelpad=15, fontsize=15)
        axes[0].set_title('Category-wise email count', pad=15, fontsize=17)
        #=================================================================================================

        #==================================================================
        categories_all = category_wise_counts.index.to_list()
        categories_counts_all = category_wise_counts.Count.to_list()
        categories_explode_all = tuple([
            0.1 if item==max(categories_counts_all) else 0 
            for item in categories_counts_all
        ])

        axes[1].pie(
            categories_counts_all,
            labels=categories_all,
            explode=categories_explode_all,
            autopct='%1.1f%%',
            shadow=False,
            startangle=100,
            textprops={'fontsize': 15}
        )
        axes[1].axis('equal')
        axes[1].set_title('Category-wise email share', pad=15, fontsize=17)
        #==================================================================

        plt.tight_layout()
        plt.subplots_adjust(left=0.1, right=0.9, wspace=0.3)
        
        if export:
            plt.savefig("category_wise_email_distribution_" + str(input_dataframe.shape[0]) + "." + ext,
                        format=ext,
                        dpi=1200,
                        bbox_inches='tight')

        plt.show();
        
    def plot_annotation_wise_frequency(self, input_dataframe: DataFrame, export: bool = False, ext: str = 'png') -> None:
        
        input_dataframe = input_dataframe.copy()
        input_dataframe['Category'] = input_dataframe['FilePath'].apply(self.get_category_from_file_path)
        
        #====================================================================================
        label_wise_counts = input_dataframe.groupby(
            'Label'
        ).agg(
            Count=pd.NamedAgg(column='FilePath', aggfunc='count')
        ).sort_values(
            by='Label', ascending=True
        )
        label_wise_counts['Percentage'] = label_wise_counts['Count']/input_dataframe.shape[0]
        label_wise_counts['Percentage'] = round(100*label_wise_counts['Percentage'], 2)
        #====================================================================================

        #=====================================================
        category_wise_counts = input_dataframe.groupby(
            'Category'
        ).agg(
            Count=pd.NamedAgg(column='Token', aggfunc='count')
        ).sort_values(
            by='Category', ascending=True
        )
        #=====================================================

        #===========================================================
        fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(15, 12))
        fig.subplots_adjust(wspace=0.05)
        fig.subplots_adjust(hspace=0.50)
        #===========================================================

        #=============================================================================================
        max_value_lwbp = label_wise_counts.Count.max()
        buffer_lwbp = max_value_lwbp * 0.25
        rightmost_tick_lwbp = max_value_lwbp + buffer_lwbp
        first_tick_lwbp = max_value_lwbp * 0.01
        ticks_lwbp = np.linspace(first_tick_lwbp, rightmost_tick_lwbp, 5)
        
        label_wise_counts.Count.plot.barh(
            ax=axes[0][0],
            color='#4DD0E1',
            alpha=0.8
        )
        
        axes[0][0].set_xlim(0, rightmost_tick_lwbp)
        axes[0][0].set_xticks(
            ticks_lwbp,
            [self.__human_format_xaxis(tick) for tick in ticks_lwbp],
            fontsize=15
        )
        axes[0][0].tick_params(axis='y', labelrotation=360, labelsize=15)
        axes[0][0].grid(visible=False, axis='y')
        axes[0][0].xaxis.grid(True, linestyle='--', alpha=0.7)
        axes[0][0].bar_label(
            axes[0][0].containers[0], 
            labels=[self.__human_format_for_bars(value) for value in label_wise_counts.Count.tolist()],
            padding=2,
            color='r',
            fontsize=13
        )
        axes[0][0].set_xlabel('Entity Count', labelpad=15, fontsize=15)
        axes[0][0].set_ylabel('Type', labelpad=15, fontsize=15)
        axes[0][0].set_title('Type-wise entity count', pad=15, fontsize=17)
        #=============================================================================================

        #=================================================================================================
        max_value_cwbp = category_wise_counts.Count.max()
        buffer_cwbp = max_value_cwbp * 0.25
        rightmost_tick_cwbp = max_value_cwbp + buffer_cwbp
        first_tick_cwbp = max_value_cwbp * 0.01
        ticks_cwbp = np.linspace(first_tick_cwbp, rightmost_tick_cwbp, 5)
        
        category_wise_counts.Count.plot.barh(
            ax=axes[1][0],
            color='#4DD0E1',
            alpha=0.8
        )
        
        axes[1][0].set_xlim(0, rightmost_tick_cwbp)
        axes[1][0].set_xticks(
            ticks_cwbp,
            [self.__human_format_xaxis(tick) for tick in ticks_cwbp],
            fontsize=15
        )
        axes[1][0].tick_params(axis='y', labelrotation=360, labelsize=15)
        axes[1][0].grid(visible=False, axis='y')
        axes[1][0].xaxis.grid(True, linestyle='--', alpha=0.7)
        axes[1][0].bar_label(
            axes[1][0].containers[0], 
            labels=[self.__human_format_for_bars(value) for value in category_wise_counts.Count.tolist()],
            padding=2,
            color='r',
            fontsize=13
        )
        axes[1][0].set_xlabel('Entity Count', labelpad=15, fontsize=15)
        axes[1][0].set_ylabel('Category', labelpad=15, fontsize=15)
        axes[1][0].set_title('Category-wise entity count', pad=15, fontsize=17)
        #=================================================================================================


        #===================================================================================================
        labels_all = label_wise_counts.index.tolist()
        labels_custom = label_wise_counts[label_wise_counts.Percentage > 5].index.tolist()
        labels_other = sorted(list(set(labels_all)-set(labels_custom)))
        labels_custom += ['+'.join(labels_other)]
        labels_counts_custom = label_wise_counts[label_wise_counts.index.isin(labels_custom)].Count.tolist()
        labels_counts_other = label_wise_counts[~label_wise_counts.index.isin(labels_custom)].Count.tolist()
        labels_counts_custom += [sum(labels_counts_other)]
        labels_explode_custom = tuple([
            0.1 if item==max(labels_counts_custom) else 0 
            for item in labels_counts_custom
        ])

        axes[0][1].pie(
            labels_counts_custom,
            labels=labels_custom,
            explode=labels_explode_custom,
            autopct='%1.1f%%',
            startangle=100,
            textprops={'fontsize': 14}
        )
        axes[0][1].axis('equal')
        axes[0][1].set_title('Type-wise entity share', pad=20, fontsize=17)
        #===================================================================================================

        #======================================================================
        categories_all = category_wise_counts.index.to_list()
        categories_counts_all = category_wise_counts.Count.to_list()
        categories_explode_all = tuple([
            0.1 if item==max(categories_counts_all) else 0 
            for item in categories_counts_all
        ])

        axes[1][1].pie(
            categories_counts_all,
            labels=categories_all,
            explode=categories_explode_all,
            autopct='%1.1f%%',
            startangle=100,
            textprops={'fontsize': 14}
        )
        axes[1][1].axis('equal')
        axes[1][1].set_title('Category-wise entity share', pad=20, fontsize=17)
        #======================================================================

        if export:
            plt.savefig("type_wise_and_category_wise_entity_distribution_" + str(input_dataframe.shape[0]) + "." + ext,
                        format=ext,
                        dpi=1200,
                        bbox_inches='tight')

        plt.show();
        
    

    # private

    def __load_config(self) -> Dict:
        with open(self.config_path_str) as config_reader:
            config = yaml.load(config_reader, Loader=SafeLoader)
            root_config = config[self.__get_config_key_root()]
        return root_config

    def __get_data_version_suffix(self) -> str:
        return ('_' + self.data_version) if self.data_version is not None and self.data_version != '' else ''

    def __get_xl_dir_name(self) -> str:
        return (
            'CodEAlltag_pXL' +
            self.__get_data_version_suffix()
        )

    def __get_xl_dir_prefix(self) -> str:
        return re.sub(self.data_version, '', self.__get_xl_dir_name())

    def __get_xl_ann_dir_name(self) -> str:
        return (
            'CodEAlltag_pXL_ann' +
            self.__get_data_version_suffix()
        )

    def __check_and_print_status_for_root_dir(self) -> None:
        emails_root_exists = self.exists_and_isdir(self.xl_dir_path)
        annotations_root_exists = self.exists_and_isdir(self.xl_ann_dir_path)
        self.print_tabular([
            [
                'EMAILS_ROOT_DIR',
                self.xl_dir_path,
                self.__get_tick() if emails_root_exists else self.__get_cross()
            ],
            [
                'ANNOTATIONS_ROOT_DIR',
                self.xl_ann_dir_path,
                self.__get_tick() if annotations_root_exists else self.__get_cross()
            ]
        ])
        print()

    def __check_and_print_status_for_content_in_root_dir(self, missing_annotations: List[str]) -> None:
        emails_root_exists = self.exists_and_isdir(self.xl_dir_path)
        annotations_root_exists = self.exists_and_isdir(self.xl_ann_dir_path)
        if emails_root_exists and annotations_root_exists:
            emails_subdir_map = {
                os.path.basename(item): item
                for item in glob.glob(os.path.join(self.xl_dir_path, self.__get_xl_dir_prefix() + '*'))
                if os.path.isdir(item)
            }
            self.__check_and_print_status_for_subdir(emails_subdir_map)
            self.__check_and_print_status_for_content_in_subdir(emails_subdir_map, missing_annotations)

    def __check_and_print_status_for_subdir(self, emails_subdir_map: Dict[str, str]) -> None:
        messages = []
        for index, subdir in tqdm(enumerate(list(emails_subdir_map.keys()))):
            annotations_subdir = os.path.join(self.xl_ann_dir_path, subdir)
            annotations_subdir_exists = self.exists_and_isdir(annotations_subdir)
            messages.append([
                str(f'[{index + 1}]'),
                subdir,
                annotations_subdir,
                self.__get_tick() if annotations_subdir_exists else self.__get_cross()
            ])
        self.print_tabular(messages)
        print()

    def __check_and_print_status_for_content_in_subdir(self, emails_subdir_map: Dict[str, str], missing: List[str]):
        total_found_emails = 0
        for index, subdir in enumerate(emails_subdir_map.keys()):
            self.print_tabular(messages=[[str(f'[{index + 1}]'), subdir]])

            found_emails_by_subdir = 0
            missing_annotations_by_subdir = 0

            emails_subdir_path = emails_subdir_map.get(subdir)
            for file_path in tqdm(
                    glob.glob(
                        os.path.join(emails_subdir_path, '**', ('*' + self.__get_file_extension_txt())),
                        recursive=True
                    )
            ):
                found_emails_by_subdir += 1
                if not os.path.exists(self.get_annotation_file_path_from_email_file_path(file_path)):
                    missing_annotations_by_subdir = missing_annotations_by_subdir + 1
                    missing.append(file_path)

            print()
            print(
                f"{str(found_emails_by_subdir):>7}"
                f"{'':<4}"
                f"{str(missing_annotations_by_subdir) if missing_annotations_by_subdir > 0 else self.__get_tick():>7}"
            )
            print()

            total_found_emails += found_emails_by_subdir

        print('-' * 50)
        print(
            f"{str(total_found_emails):>7}"
            f"{'':<4}"
            f"{str(len(missing)) if len(missing) > 0 else self.__get_tick():>7}"
        )

    def __read_file(self, file_path: str, show: bool = False) -> str:
        with open(file_path, encoding=self.__get_encoding_utf8()) as reader:
            content = reader.read()

        if show:
            print(file_path)
            print(self.__get_hyphen() * len(file_path) + self.__get_newline())
            print(content)
            print()

        return content

    def __get_category_path_df_file_name(self) -> str:
        return (
            'CategoryPath_DF' +
            self.__get_data_version_suffix() +
            '.csv'
        )

    def __get_category_path_df_file_path(self) -> str:
        return os.path.join(self.xl_dir_path, self.__get_category_path_df_file_name())

    def __get_annotation_df_file_name(self) -> str:
        return (
            'Annotation_DF' +
            self.__get_data_version_suffix() +
            '.csv'
        )

    def __get_annotation_df_file_path(self) -> str:
        return os.path.join(self.xl_ann_dir_path, self.__get_annotation_df_file_name())

    def __create_category_path_df(self) -> None:
        category_path_tuples = [
            (
                file_path.split(self.xl_dir_path)[1].split(os.sep)[1].split(self.__get_xl_dir_prefix())[1],
                file_path.replace(self.xl_dir_path + os.sep, ''),
                os.path.getsize(file_path),
                os.path.exists(self.get_annotation_file_path_from_email_file_path(file_path))
            ) for file_path in tqdm(
                glob.glob(
                    os.path.join(self.xl_dir_path, '**', ('*' + self.__get_file_extension_txt())),
                    recursive=True
                ),
                position=0,
                leave=True
            )
        ]
        df = pd.DataFrame(category_path_tuples, columns=self.__get_category_path_df_columns())
        df.to_csv(self.__get_category_path_df_file_path())

    def __create_annotation_df(self, use_only_paths: List[str] = None) -> None:
        if use_only_paths is None or not use_only_paths:
            file_paths = self.get_category_path_df()['FilePath'].tolist()
        else:
            file_paths = use_only_paths

        annotation_tuples = list()
        with tqdm(total=len(file_paths), position=0, leave=True) as progress_bar:
            for file_path in file_paths:
                annotation_tuples_by_file = self.get_annotation_tuples_by_file(file_path)
                if len(annotation_tuples_by_file) > 0:
                    annotation_tuples += annotation_tuples_by_file
                progress_bar.update(1)

        df = pd.DataFrame(annotation_tuples, columns=self.__get_annotation_df_columns())
        df.to_csv(self.__get_annotation_df_file_path())
    
    def __get_same_entity_labels_path_df_path(self, reference_cdp: CodealltagDataProcessor) -> str:
        file_name = (
            'SameEntityLabelsPath_DF_' + 
            self.get_data_version() + 
            '_'  + 
            reference_cdp.get_data_version() + 
            '_' +
            str(self.max_file_size) + 
            '.csv'
        )
        return os.path.join(self.get_xl_dir_path(), file_name)
    
    def __create_same_entity_labels_path_df(self, reference_cdp: CodealltagDataProcessor) -> None:
        
        category_path_df = self.get_category_path_df()
        same_entity_labels_path_df_tuples = list()
        total_files_with_same_entity_labels = 0
        
        # [1]
        # filter files that have file size <= 1000 bytes
        category_path_df_filtered = category_path_df[category_path_df.FileSize <= self.max_file_size]
        
        category_path_df_filtered_rows = category_path_df_filtered.itertuples(index=True)
        with tqdm(total=len(category_path_df_filtered), position=0, leave=True) as progress_bar:
            for row in category_path_df_filtered_rows:
                
                email_file_path_reference_exists = os.path.exists(
                    reference_cdp.get_absolute_email_file_path(row.FilePath)
                )
                annotations_file_path_exists = os.path.exists(
                    self.get_absolute_annotations_file_path(row.FilePath)
                )
                annotations_file_path_reference_exists = os.path.exists(
                    reference_cdp.get_absolute_annotations_file_path(row.FilePath)
                )
                
                # [2]
                # filter files that are present in both versions
                if email_file_path_reference_exists and annotations_file_path_exists and annotations_file_path_reference_exists:
                    
                    annotation_df_by_file = self.get_annotation_df_by_file(row.FilePath)
                    annotation_df_by_file_reference = reference_cdp.get_annotation_df_by_file(row.FilePath)
                    
                    # [3]
                    # filter files that have at least one entity in respective annotation files
                    if not annotation_df_by_file.empty and not annotation_df_by_file_reference.empty:
                        
                        # [4]
                        # filter files that have same number of entities in both versions
                        if len(annotation_df_by_file) == len(annotation_df_by_file_reference):
                            
                            labels = annotation_df_by_file.Label.str.cat(sep='-')
                            labels_reference = annotation_df_by_file_reference.Label.str.cat(sep='-')
                            
                            # [5]
                            # filter files that have exact same entity labels:
                            if labels == labels_reference:
                                
                                same_entity_labels_path_df_tuples.append((
                                    row.Index,
                                    row.Category,
                                    row.FilePath,
                                    row.FileSize,
                                    row.AnnotationFileExists
                                ))
                                total_files_with_same_entity_labels += 1
                            
                progress_bar.update(1)
        
        df = pd.DataFrame(
            same_entity_labels_path_df_tuples,
            columns=["ID"] + self.__get_category_path_df_columns()
        )
        df.to_csv(self.__get_same_entity_labels_path_df_path(reference_cdp))
    
    def __get_same_entity_labels_path_df_path_for_sample_size(self, 
                                                              reference_cdp: CodealltagDataProcessor,
                                                              sample_size: int) -> str:
        file_name = (
            'SameEntityLabelsPath_DF_' + 
            self.get_data_version() + 
            '_'  + 
            reference_cdp.get_data_version() + 
            '_' +
            str(self.max_file_size) + 
            '_' + 
            str(sample_size // 1000) + 'K' +
            '.csv'
        )
        return os.path.join(self.get_xl_dir_path(), file_name)
        
    def __prepare_3K_10K_samples(self, reference_cdp: CodealltagDataProcessor) -> None:
        
        same_entity_labels_path_df = self.get_same_entity_labels_path_df(reference_cdp=reference_cdp)
        lrl_info_dict = self.__prepare_low_representative_label_info_dict(same_entity_labels_path_df)
        
        for sample_size in range(3000, 10000 + 1, 1000):
            
            sample_ids: List[int] = []
            
            lrl_ids_for_sample_size = self.__get_low_representative_label_ids_for_sample_size(
                sample_size=sample_size,
                low_representative_label_info_dict=lrl_info_dict
            )
            sample_ids.extend(lrl_ids_for_sample_size)
            
            cwrf_for_sample_size = self.__calculate_category_wise_required_files_for_sample_size(
                sample_size=sample_size,
                ids_for_sample_size=lrl_ids_for_sample_size,
                same_entity_labels_path_df=same_entity_labels_path_df
            )
            cw_ids_for_sample_size = self.__get_category_wise_ids_for_sample_size(
                sample_size=sample_size,
                category_wise_required_files=cwrf_for_sample_size,
                low_representative_ids=lrl_info_dict['IDS'],
                same_entity_labels_path_df=same_entity_labels_path_df
            )
            sample_ids.extend(cw_ids_for_sample_size)
            
            sample_df = same_entity_labels_path_df[same_entity_labels_path_df.ID.isin(sample_ids)]
            sample_df.to_csv(self.__get_same_entity_labels_path_df_path_for_sample_size(reference_cdp, sample_size))
    

    def __prepare_low_representative_label_info_dict(self, 
                                                     same_entity_labels_path_df: DataFrame, 
                                                     threshold: float = 0.025):
        
        low_representative_label_info_dict: Dict[str, Any] = dict()
        
        annotation_df = self.get_annotation_df()
        annotation_df_filtered = annotation_df[
            annotation_df.FilePath.isin(
                same_entity_labels_path_df.FilePath.str.replace('.txt', '.ann', regex=True)
            )
        ]
        
        label_wise_ratios = self.get_category_or_label_wise_count_or_ratio(
            input_dataframe=annotation_df_filtered,
            category_wise=False,
            ratio=True,
            precision=5
        )
        
        low_representative_labels: List[str] = []
        for key, value in label_wise_ratios.items():
            if value < threshold:
                low_representative_labels.append(key)
                low_representative_label_info_dict[key+'_RATIO'] = value
        
        low_representative_label_info_dict['LABELS'] = low_representative_labels
        
        
        adf_txt = annotation_df_filtered.copy()
        adf_txt['FilePath'] = adf_txt.FilePath.replace('.ann', '.txt', regex=True)
        for label in low_representative_labels:
            selpd_frac = same_entity_labels_path_df[
                same_entity_labels_path_df.FilePath.isin(adf_txt[adf_txt.Label==label].FilePath)
            ]
            low_representative_label_info_dict[label+'_IDS'] = selpd_frac.ID.to_list()
            low_representative_label_info_dict[label+'_MLA'] = selpd_frac.shape[0] // (10 - 3 + 1)
        
        low_representative_label_ids: List[int] = same_entity_labels_path_df[
                same_entity_labels_path_df.FilePath.isin(
                    adf_txt[adf_txt.Label.isin(low_representative_labels)].FilePath
                )
        ].ID.unique().tolist()
        
        low_representative_label_info_dict['IDS'] = low_representative_label_ids
        return low_representative_label_info_dict

    
    def __get_low_representative_label_ids_for_sample_size(self, 
                                                           sample_size: int,
                                                           low_representative_label_info_dict: Dict[str, Any]) -> List[int]:
        ids: Set[int] = set()
        multiplier = int(((sample_size - 3000) // 1000) + 1)
        for label in low_representative_label_info_dict['LABELS']:
            ids = ids.union(
                set(
                    low_representative_label_info_dict[label+'_IDS'][
                        0:(low_representative_label_info_dict[label+'_MLA']*multiplier)
                    ]
                )
            )
        return list(ids)
    
    def __calculate_category_wise_required_files_for_sample_size(self,
                                                                 sample_size: int,
                                                                 ids_for_sample_size: List[int],
                                                                 same_entity_labels_path_df: DataFrame) -> Dict[str, int]:
        
        category_wise_ratio = self.get_category_or_label_wise_count_or_ratio(
            input_dataframe=same_entity_labels_path_df,
            category_wise=True,
            ratio=True,
            precision=2
        )
        category_wise_required_files = {
            key: int(sample_size*value) 
            for key, value in category_wise_ratio.items()
        }
        
        category_wise_existing_files = self.get_category_or_label_wise_count_or_ratio(
            input_dataframe=same_entity_labels_path_df[
                same_entity_labels_path_df.ID.isin(ids_for_sample_size)
            ]
        )
        category_wise_extra_files = {
            key: (value - category_wise_required_files[key])
            for key, value in category_wise_existing_files.items()
            if value > category_wise_required_files[key]
        }
        
        penalty_per_category: Dict[str, int] = {}
        if sum(category_wise_extra_files.values()) > 0:
            category_wise_required_files_tmp = {
                key: (value - (category_wise_existing_files[key] if key in category_wise_existing_files.keys() else 0))
                for key, value in category_wise_required_files.items()
                if key not in category_wise_extra_files.keys()
            }
            
            adjusted_category_wise_ratio = {
                key: (value / sum(category_wise_required_files_tmp.values()))                            
                for key, value in category_wise_required_files_tmp.items()
            }
            
            penalty_per_category = {
                key: int(round(sum(category_wise_extra_files.values()) * value, 0))
                for key, value in adjusted_category_wise_ratio.items()
            }
        
        adjusted_category_wise_required_files: Dict[str, int] = {}
        for key, value in category_wise_required_files.items():
            if key not in category_wise_extra_files:
                adjusted_value = value
                if key in category_wise_existing_files:
                    adjusted_value = adjusted_value - category_wise_existing_files[key]
                if key in penalty_per_category:
                    adjusted_value = adjusted_value - penalty_per_category[key]
                adjusted_category_wise_required_files[key] = adjusted_value
        
        total = sum(category_wise_existing_files.values()) + sum(adjusted_category_wise_required_files.values())
        if total > sample_size:
            max_value = max(adjusted_category_wise_required_files.values())
            keys_with_max = [
                key 
                for key in adjusted_category_wise_required_files 
                if adjusted_category_wise_required_files[key] == max_value
            ]
            key_with_max = keys_with_max[0]
            adjusted_category_wise_required_files[key_with_max] = (
                adjusted_category_wise_required_files[key_with_max] - (total - sample_size)
            )
        elif total < sample_size:
            min_value = min(adjusted_category_wise_required_files.values())
            keys_with_min = [
                key 
                for key in adjusted_category_wise_required_files 
                if adjusted_category_wise_required_files[key] == min_value
            ]
            key_with_min = keys_with_min[0]
            adjusted_category_wise_required_files[key_with_min] = (
                adjusted_category_wise_required_files[key_with_min] + (sample_size - total)
            )
        
        return adjusted_category_wise_required_files
    
    
    def __get_category_wise_ids_for_sample_size(self,
                                                sample_size: int, 
                                                category_wise_required_files: Dict[str, int], 
                                                low_representative_ids: List[int],
                                                same_entity_labels_path_df: DataFrame):

        category_wise_ids: List[int] = []
        same_entity_labels_path_df_wo_lri = same_entity_labels_path_df[
            ~same_entity_labels_path_df.ID.isin(low_representative_ids)
        ]
        for key in category_wise_required_files.keys():
            ids_by_cat = same_entity_labels_path_df_wo_lri[same_entity_labels_path_df_wo_lri.Category==key]['ID'].tolist()
            random.seed(sample_size)
            selected_ids_by_cat = random.sample(ids_by_cat, category_wise_required_files[key])
            category_wise_ids.extend(selected_ids_by_cat)

        return category_wise_ids
    
    def __prepare_model_data_for_sample_size(self,
                                             reference_cdp: CodealltagDataProcessor,
                                             sample_size: int) -> DataFrame:
        
        print(f'[START] prepare model data for sample size: {sample_size}\n')
        sample_df = self.get_same_entity_labels_path_df_for_sample_size(reference_cdp, sample_size)
        
        inp_type_1_texts: List[str] = list()
        inp_type_2_texts: List[str] = list()
        out_type_1_texts: List[str] = list()
        out_type_2_texts: List[str] = list()
        
        with tqdm(total=sample_df.shape[0], position=0, leave=True) as progress_bar:
            for index, row in sample_df.iterrows():
                email_text = self.read_email(email_file_path)[1]
                annotation_df = self.get_annotation_df_by_file(email_file_path)
                
                inp_type_1_text = self.tokenize_with_somajo_and_annotation_df(email_text, annotation_df)
                inp_type_1_texts.append(inp_type_1_text)

                inp_type_2_text = " ".join(self.tokenize_with_somajo(email_text))
                inp_type_2_texts.append(inp_type_2_text)

                out_type_1_text = self.get_output_sequence_for_seq2seq_ner_model(row.FilePath)
                out_type_1_texts.append(out_type_1_text)

                out_type_2_text = self.get_output_sequence_for_seq2seq_ner_pg_model(row.FilePath, reference_cdp)
                out_type_2_texts.append(out_type_2_text)
                
                progress_bar.update(1)
                
        sample_df["InputType1"] = inp_type_1_texts
        sample_df["InputType2"] = inp_type_2_texts
        sample_df["OutputType1"] = out_type_1_texts
        sample_df["OutputType2"] = out_type_2_texts
        
        sample_df.to_csv(self.__get_same_entity_labels_path_df_path_for_sample_size(reference_cdp, sample_size))
        
        print(f'[END] prepare model data for sample size: {sample_size}\n')
        return sample_df
    
    def __clean_dir(self, dir_path: str) -> None:
        if dir_path.startswith(self.data_dir_path):
            for path in Path(dir_path).glob("**/*"):
                if path.is_file():
                    path.unlink()
                elif path.is_dir():
                    rmtree(path)
    
    # static-public

    @staticmethod
    def print_tabular(messages: List[List[str]], tab_size: int = 4) -> None:
        column_wise_max_lengths = [
            max([len(row) for row in columns])
            for columns in list(zip(*messages))
        ]
        total_columns = len(column_wise_max_lengths)
        format_template = '%%%ds' * total_columns
        formatter = format_template % tuple([
            -(max_length + tab_size if index + 1 < total_columns else max_length)
            for index, max_length in enumerate(column_wise_max_lengths)
        ])
        for message in messages:
            print(formatter % tuple(message))

    @staticmethod
    def exists_and_isdir(path: str) -> bool:
        return os.path.exists(path) and os.path.isdir(path)
    
    @staticmethod
    def get_category_or_label_wise_count_or_ratio(input_dataframe: DataFrame,
                                                  category_wise: bool = True,
                                                  ratio: bool = False,
                                                  precision: int = 2) -> Dict[str, Union[int, float]]:
    
        aggregation_df = input_dataframe.groupby(
            ('Category' if category_wise else 'Label')
        ).agg(
            Count=pd.NamedAgg(column='FilePath', aggfunc='count')
        ).sort_values(
            by='Count', ascending=True
        )
        if ratio:
            aggregation_df['Ratio'] = round(aggregation_df['Count']/input_dataframe.shape[0], precision)

        category_wise_count_or_ratio: Dict[str, Union[int, float]] = {}
        for key, value in zip(aggregation_df.index, aggregation_df[('Ratio' if ratio else 'Count')]):
            category_wise_count_or_ratio[key] = value

        return category_wise_count_or_ratio
    
    @staticmethod
    def get_train_dev_folds(n_fold: int = 5) -> List[Tuple]:
        fold_tuples = list()
        indices = list(range(n_fold))
        for index in indices:
            rolled_indices = np.roll(indices, -index)
            train_indices = list(rolled_indices[0:(n_fold - 1)])
            dev_indices = [rolled_indices[-1]]
            fold_tuples.append((
                index + 1,
                train_indices,
                dev_indices
            ))
        return fold_tuples
    
    @staticmethod
    def get_annotation_df_with_input_text_and_predicted_text(input_text: str, 
                                                             predicted_text: str,
                                                             labels: List[str]) -> DataFrame:
        tuples = list()

        input_text_length = len(input_text)
        input_text_copy = input_text[0: input_text_length]

        item_delim = "; "
        token_delim = ": "
        pseudonym_delim = " **"
        token_id = 0
        next_cursor = 0

        predicted_items = predicted_text.split(item_delim)
        for item in predicted_items:

            label, token, pseudonym = "", "", ""

            for l in labels:
                if item.startswith(l):
                    label = l

            if label != "" and (label+token_delim) in item:
                
                value_splits = item.split(label+token_delim)
                token_pseudonym = value_splits[1]

                if (pseudonym_delim in token_pseudonym and token_pseudonym.endswith(pseudonym_delim.strip())):

                    pseudonym_splits = token_pseudonym.split(pseudonym_delim)
                    token = pseudonym_splits[0]
                    pseudonym = pseudonym_splits[1][:-2]

                else:
                    token = token_pseudonym
                
                if len(token.strip()) > 0:

                    start = input_text_copy.find(token)
                    if start == -1 and ' ' in token:
                        start = input_text_copy.find(token.split(' ')[0])
                        token = token.replace(' ', '')

                    if start != -1:
                        end = start + len(token)

                        token_id += 1
                        prev_cursor = next_cursor
                        next_cursor += end
                        input_text_copy = input_text[next_cursor: input_text_length]

                        start = prev_cursor + start
                        end = prev_cursor + end

                        tuples.append((
                            'T' + str(token_id),
                            label,
                            start,
                            end,
                            input_text[start:end],
                            pseudonym
                        ))

        return pd.DataFrame(
            tuples,
            columns=["Token_ID", "Label", "Start", "End", "Token", "Pseudonym"]
        )
    
    @staticmethod
    def get_token_label_tuples(text: str) -> List[Tuple[str, str]]:
        lines = text.strip().split("\n")  # Split input by new lines
        result = [tuple(line.rsplit(" ", 1)) for line in lines]  # Split each line into a tuple (word, tag)
        return result
    
    @staticmethod
    def align_tags(list_a: List[Tuple[str, str]], list_b: List[Tuple[str, str]]) -> List[str]:
        result: List[str] = list()
        idx_b = 0  # Pointer for tracking position in list_b

        buffer: str = ""  # To accumulate tokens for merging

        for token_a, _ in list_a:
            if idx_b < len(list_b) and token_a == list_b[idx_b][0]:
                # If tokens match, take the tag from list_b
                result.append(list_b[idx_b][1])
                idx_b += 1  # Move to the next index in list_b
                buffer = ""  # Reset buffer
            else:
                # If tokens don't match, add 'O' to the result
                result.append('O')
                buffer += token_a  # Merge tokens

                # Check if accumulated tokens now match a token in list_b
                if idx_b < len(list_b) and buffer == list_b[idx_b][0]:
                    result[-1] = list_b[idx_b][1]  # Assign correct tag from list_b
                    idx_b += 1  # Move to the next token in list_b
                    buffer = ""  # Reset buffer after a match

        return result
    
    @staticmethod
    def get_pseudonymized_text(input_text: str, predicted_annotation_df: DataFrame) -> str:
        output_text = input_text
        offset = 0
        for index, row in predicted_annotation_df.iterrows():
            output_text = output_text[:(row.Start+offset)] + row.Pseudonym + output_text[(row.End+offset):]
            offset += len(row.Pseudonym) - len(row.Token)
        return output_text

    @staticmethod
    def cuda_empty_cache() -> None:
        torch.cuda.empty_cache()

    @staticmethod
    def print_cuda_memory_summary() -> None:
        print(torch.cuda.memory_summary(device=None, abbreviated=False))


    # static-private

    @staticmethod
    def __get_config_key_root() -> str:
        return 'codealltag_data_processor'

    @staticmethod
    def __get_config_key_random_seed() -> str:
        return 'random_seed'

    @staticmethod
    def __get_config_key_data_dir_path() -> str:
        return 'data_dir_path'

    @staticmethod
    def __get_config_key_max_file_size() -> str:
        return 'max_file_size'

    @staticmethod
    def __get_file_extension_txt() -> str:
        return '.txt'

    @staticmethod
    def __get_file_extension_ann() -> str:
        return '.ann'
    
    @staticmethod
    def __get_file_extension_csv() -> str:
        return '.csv'

    @staticmethod
    def __get_encoding_utf8() -> str:
        return 'utf-8'

    @staticmethod
    def __get_newline() -> str:
        return '\n'

    @staticmethod
    def __get_hyphen() -> str:
        return '-'

    @staticmethod
    def __get_whitespace_regex() -> str:
        return r'\s+'

    @staticmethod
    def __get_tick() -> str:
        return '\u2713'

    @staticmethod
    def __get_cross() -> str:
        return '\u274C'

    @staticmethod
    def __get_category_path_df_columns() -> List[str]:
        return ['Category', 'FilePath', 'FileSize', 'AnnotationFileExists']

    @staticmethod
    def __get_annotation_df_columns() -> List[str]:
        return ['FilePath', 'Token_ID', 'Label', 'Start', 'End', 'Token']
    
    @staticmethod
    def __human_format_for_bars(x):
        if x >= 1_000_000:
            return f'{x / 1_000_000:.2f}M'
        elif x >= 1_000:
            return f'{x / 1_000:.2f}K'
        return str(int(x))
    
    @staticmethod
    def __human_format_xaxis(x):
        if x >= 1_000_000:
            return f'{x / 1_000_000:.0f}M'
        elif x >= 1_000:
            return f'{x / 1_000:.0f}K'
        return str(int(x))
    
    @staticmethod
    def __get_somajo_tokenizer() -> SoMaJo:
        return SoMaJo("de_CMC", split_camel_case=False)


    # property

    data_version = property(get_data_version)
    config_path_str = property(get_config_path_str)
    random_seed = property(get_random_seed)
    data_dir_path = property(get_data_dir_path)
    max_file_size = property(get_max_file_size)
    xl_dir_path = property(get_xl_dir_path)
    xl_ann_dir_path = property(get_xl_ann_dir_path)