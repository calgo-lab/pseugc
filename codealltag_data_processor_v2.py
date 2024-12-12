from __future__ import annotations
from datasets import Dataset, DatasetDict
import glob
import os
import re
from pathlib import Path
from shutil import copy, rmtree
from typing import Tuple, List, Dict, Union, Any

import numpy as np
import pandas as pd
import torch
import yaml
from io import StringIO
from IPython.display import display
from pandas import DataFrame
from pandas.core.indexes.numeric import Int64Index
from sklearn.model_selection import KFold
from somajo import SoMaJo
from tqdm import tqdm
from yaml.loader import SafeLoader


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

    def get_tokenizer_version(self) -> str:
        return self.__config[self.__get_config_key_tokenizer_version()]

    def get_max_file_count(self) -> int:
        return self.__config[self.__get_config_key_max_file_count()]

    def get_max_file_size(self) -> int:
        return self.__config[self.__get_config_key_max_file_size()]

    def get_multiplier(self) -> int:
        return self.__config[self.__get_config_key_multiplier()]

    def get_test_file_count(self) -> int:
        return self.__config[self.__get_config_key_test_file_count()]

    def get_k_fold(self) -> int:
        return self.__config[self.__get_config_key_k_fold()]

    def get_use_fold(self) -> int:
        return self.__config[self.__get_config_key_use_fold()]

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

    def select_category_path_df(self,
                                max_file_count: int,
                                max_file_size: int,
                                multiplier: int = 5,
                                use_only_indices: Int64Index = None) -> DataFrame:

        category_path_df = self.get_category_path_df()

        if max_file_size is None:
            max_file_size = category_path_df.FileSize.max()

        print(f'selecting files ... ['
              f'max_file_count={max_file_count}, '
              f'max_file_size={max_file_size}, '
              f'multiplier={multiplier}'
              f'] ')

        if use_only_indices is not None:
            category_path_df_selected = category_path_df.loc[use_only_indices].copy()
        else:
            category_path_df_selected = category_path_df.copy()

        category_path_df_filtered = category_path_df_selected[
            category_path_df_selected.FileSize <= max_file_size
        ].copy()

        category_path_df_rows = category_path_df_filtered.sample(
            n=max_file_count * multiplier,
            random_state=self.random_seed
        ).itertuples(index=True)

        selected_category_path_tuples = list()
        selected_file_count = 0

        with tqdm(total=max_file_count, position=0, leave=True) as progress_bar:
            for row in category_path_df_rows:
                annotation_df_by_file = self.get_annotation_df_by_file(row.FilePath)
                if not annotation_df_by_file.empty:
                    selected_category_path_tuples.append((
                        row.Index,
                        row.Category,
                        row.FilePath,
                        row.FileSize,
                        None
                    ))
                    selected_file_count += 1
                    progress_bar.update(1)

                if selected_file_count == max_file_count:
                    break

        print()

        return pd.DataFrame(selected_category_path_tuples, columns=self.__get_selected_category_path_df_columns())

    def get_selected_category_path_df(self, force_create: bool = False) -> DataFrame:
        selected_category_path_df_path = self.__get_selected_category_path_df_path()
        if not os.path.exists(selected_category_path_df_path) or force_create:
            self.__create_selected_file_path_df()
        return pd.read_csv(selected_category_path_df_path, index_col=0)

    def copy_email_and_annotation_files_to_dir(self,
                                               selected_category_path_df: DataFrame,
                                               dir_name_prefix: str) -> None:
        dir_path = os.path.join(self.data_dir_path, dir_name_prefix + self.__get_data_version_suffix())
        file_paths = selected_category_path_df['FilePath'].tolist()

        with tqdm(total=len(file_paths), position=0, leave=True) as progress_bar:
            for file_path in file_paths:
                file_dir_path = os.path.join(dir_path, os.path.dirname(file_path))
                os.makedirs(file_dir_path, exist_ok=True)

                email_file_path = os.path.join(self.xl_dir_path, file_path)
                annotation_file_path = self.get_annotation_file_path_from_email_file_path(email_file_path)

                if os.path.exists(email_file_path) and os.path.exists(annotation_file_path):
                    copy(email_file_path, file_dir_path)
                    copy(annotation_file_path, file_dir_path)

                progress_bar.update(1)
    
    
    def get_missed_annotation_df(self) -> DataFrame:
        return pd.read_csv(
            self.__get_missed_annotation_df_path(),
            index_col=0
        )

    def get_train_dev_k_folds(self) -> List[Tuple]:
        fold_tuples = list()
        indices = list(range(self.k_fold))
        for index in indices:
            rolled_indices = np.roll(indices, -index)
            train_indices = list(rolled_indices[0:(self.k_fold - 1)])
            dev_indices = [rolled_indices[-1]]
            fold_tuples.append((
                index + 1,
                train_indices,
                dev_indices
            ))
        return fold_tuples

    def get_train_dev_k_folds_indices(self) -> List[Tuple]:
        fold_tuples = list()
        selected_file_path_df = self.get_selected_category_path_df().iloc[:-self.test_file_count]
        splits = list(KFold(n_splits=self.k_fold).split(selected_file_path_df.index.to_numpy()))
        train_dev_k_folds = self.get_train_dev_k_folds()
        for index, fold in enumerate(train_dev_k_folds):
            train_indices = list()
            fold_train_indices = fold[1]
            for fold_train_index in fold_train_indices:
                train_indices += list(splits[fold_train_index][1])
            dev_indices = list(splits[fold[2][0]][1])
            fold_tuples.append((
                index + 1,
                train_indices,
                dev_indices
            ))
        return fold_tuples
    
    def prepare_train_dev_test_k_folds_data_partitions(self) -> None:
        data_exists = False

        partitions_dir_path = self.__get_partitions_dir_path()
        dir_exists = self.exists_and_isdir(partitions_dir_path)

        if dir_exists:
            found_file_count = len(
                glob.glob(
                    os.path.join(partitions_dir_path, '**', ('partition' + self.__get_file_extension_csv())),
                    recursive=True
                )
            )
            if found_file_count >= self.k_fold:
                data_exists = True
        else:
            os.makedirs(partitions_dir_path, exist_ok=True)

        if data_exists:
            print('processed partitions already exists ...')
            print()
        else:
            self.__clean_dir(partitions_dir_path)

            selected_category_path_df = self.get_selected_category_path_df()

            print('processing partitions ...')
            fold_tuples = self.get_train_dev_k_folds_indices()
            with tqdm(total=len(fold_tuples), position=0, leave=True) as progress_bar:
                for fold in fold_tuples:
                    k = fold[0]

                    # 1
                    fold_dir = self.__get_train_dev_test_k_fold_dir_path(k)
                    os.makedirs(fold_dir, exist_ok=True)

                    # 2
                    fold_df = selected_category_path_df.copy()
                    fold_df.iloc[fold[1], [4]] = 'train'
                    fold_df.iloc[fold[2], [4]] = 'dev'
                    fold_df.iloc[-self.test_file_count:, [4]] = 'test'
                    fold_df.to_csv(
                        self.__get_train_dev_test_k_fold_partition_df_path(k)
                    )

                    progress_bar.update(1)

            print()

    def get_train_dev_test_k_fold_partition_df(self, k: int) -> DataFrame:
        return pd.read_csv(
            self.__get_train_dev_test_k_fold_partition_df_path(k),
            index_col=0
        )
    
    def get_somajo_split_sentences(self, email_file_path: Union[str, StringIO]) -> List[str]:
        sentences = []
        tokenizer = self.__get_somajo_tokenizer()
        if isinstance(email_file_path, str):
            email_file_path = self.get_absolute_email_file_path(email_file_path)
        
        for sentence in tokenizer.tokenize_text_file(email_file_path, paragraph_separator="empty_lines"):
            sentences.append(" ".join([token.text for token in sentence]))
        return sentences
    
    def get_selected_seq2seq_df(self, force_create: bool = False) -> DataFrame:
        selected_seq2seq_df_path = self.__get_selected_seq2seq_df_path()
        if not os.path.exists(selected_seq2seq_df_path) or force_create:
            self.__create_selected_seq2seq_df()
        return pd.read_csv(selected_seq2seq_df_path, index_col=0)
    
    def get_selected_seq2seq_datasetdict(self, k: int = 1) -> DatasetDict:
        self.prepare_train_dev_test_k_folds_data_partitions()
        seq2seq_df = self.get_selected_seq2seq_df()
        kth_partition_df = self.get_train_dev_test_k_fold_partition_df(k)
        
        df = kth_partition_df.merge(seq2seq_df, how='inner', on='ID')
        
        train_ds = Dataset.from_pandas(df[df.Group=='train'])
        dev_ds = Dataset.from_pandas(df[df.Group=='dev'])
        test_ds = Dataset.from_pandas(df[df.Group=='test'])

        return DatasetDict({
            'train': train_ds,
            'dev': dev_ds,
            'test': test_ds
        })
    
    def get_same_annotation_count_path_df(self, reference_cdp: CodealltagDataProcessor, force_create: bool = False) -> DataFrame:
        same_annotation_count_path_df_path = self.__get_same_annotation_count_path_df_path(reference_cdp)
        if not os.path.exists(same_annotation_count_path_df_path) or force_create:
            self.__create_same_annotation_count_path_df(reference_cdp)
        return pd.read_csv(same_annotation_count_path_df_path, index_col=0)
    
    def get_same_entity_labels_path_df(self, reference_cdp: CodealltagDataProcessor, force_create: bool = False) -> DataFrame:
        same_entity_labels_path_df_path = self.__get_same_entity_labels_path_df_path(reference_cdp)
        if not os.path.exists(same_entity_labels_path_df_path) or force_create:
            self.__create_same_entity_labels_path_df(reference_cdp)
        return pd.read_csv(same_entity_labels_path_df_path, index_col=0)
    
    def select_same_annotation_count_path_df(self,
                                             reference_cdp: CodealltagDataProcessor,
                                             max_file_count: int,
                                             max_file_size: int,
                                             multiplier: int = 5) -> DataFrame:

        category_path_df = self.get_category_path_df()
        
        same_annotation_count_path_df = self.get_same_annotation_count_path_df(reference_cdp)
        
        category_path_df["ID"] = category_path_df.index
        merged_df = pd.merge(same_annotation_count_path_df, category_path_df, on="FilePath", how="left")
        
        if max_file_size is None:
            max_file_size = merged_df.FileSize.max()

        print(f'selecting files ... ['
              f'reference_data_version={reference_cdp.get_data_version()}, '
              f'max_file_count={max_file_count}, '
              f'max_file_size={max_file_size}, '
              f'multiplier={multiplier}'
              f'] ')

        merged_df_filtered = merged_df[merged_df.FileSize <= max_file_size]

        merged_df_rows = merged_df_filtered.sample(
            n=max_file_count * multiplier,
            random_state=self.random_seed
        ).itertuples(index=True)

        selected_same_annotation_count_path_tuples = list()
        selected_file_count = 0

        with tqdm(total=max_file_count, position=0, leave=True) as progress_bar:
            for row in merged_df_rows:
                
                annotation_df_by_file = self.get_annotation_df_by_file(row.FilePath)
                labels = annotation_df_by_file.Label.str.cat(sep='-')
                
                annotation_df_reference_by_file = reference_cdp.get_annotation_df_by_file(row.FilePath)
                labels_reference = annotation_df_reference_by_file.Label.str.cat(sep='-')
                
                if not annotation_df_by_file.empty and labels == labels_reference:
                    selected_same_annotation_count_path_tuples.append((
                        row.ID,
                        row.Category,
                        row.FilePath,
                        row.FileSize,
                        None
                    ))
                    selected_file_count += 1
                    progress_bar.update(1)

                if selected_file_count == max_file_count:
                    break

        print()

        return pd.DataFrame(
            selected_same_annotation_count_path_tuples,
            columns=self.__get_selected_category_path_df_columns()
        )
    
    def get_selected_same_annotation_count_path_df(self, 
                                                   reference_cdp: CodealltagDataProcessor,
                                                   force_create: bool = False) -> DataFrame:
        
        df_path = self.__get_selected_same_annotation_count_path_df_path(reference_cdp)
        if not os.path.exists(df_path) or force_create:
            self.__create_selected_same_annotation_count_path_df(reference_cdp)
        return pd.read_csv(df_path, index_col=0)
    
    def get_train_dev_k_folds_indices_2(self, reference_cdp: CodealltagDataProcessor) -> List[Tuple]:
        fold_tuples = list()
        path_df = self.get_selected_same_annotation_count_path_df(reference_cdp).iloc[:-self.test_file_count]
        splits = list(KFold(n_splits=self.k_fold).split(path_df.index.to_numpy()))
        train_dev_k_folds = self.get_train_dev_k_folds()
        for index, fold in enumerate(train_dev_k_folds):
            train_indices = list()
            fold_train_indices = fold[1]
            for fold_train_index in fold_train_indices:
                train_indices += list(splits[fold_train_index][1])
            dev_indices = list(splits[fold[2][0]][1])
            fold_tuples.append((
                index + 1,
                train_indices,
                dev_indices
            ))
        return fold_tuples
    
    def prepare_train_dev_test_k_folds_data_partitions_2(self, reference_cdp: CodealltagDataProcessor) -> None:
        data_exists = False

        partitions_dir_path = self.__get_partitions_dir_path_2(reference_cdp)
        dir_exists = self.exists_and_isdir(partitions_dir_path)

        if dir_exists:
            found_file_count = len(
                glob.glob(
                    os.path.join(partitions_dir_path, '**', ('partition' + self.__get_file_extension_csv())),
                    recursive=True
                )
            )
            if found_file_count >= self.k_fold:
                data_exists = True
        else:
            os.makedirs(partitions_dir_path, exist_ok=True)

        if data_exists:
            print('processed partitions already exists ...')
            print()
        else:
            self.__clean_dir(partitions_dir_path)

            selected_df = self.get_selected_same_annotation_count_path_df(reference_cdp)

            print('processing partitions ...')
            fold_tuples = self.get_train_dev_k_folds_indices_2(reference_cdp)
            with tqdm(total=len(fold_tuples), position=0, leave=True) as progress_bar:
                for fold in fold_tuples:
                    k = fold[0]

                    # 1
                    fold_dir = self.__get_train_dev_test_k_fold_dir_path_2(reference_cdp, k)
                    os.makedirs(fold_dir, exist_ok=True)

                    # 2
                    fold_df = selected_df.copy()
                    fold_df.iloc[fold[1], [4]] = 'train'
                    fold_df.iloc[fold[2], [4]] = 'dev'
                    fold_df.iloc[-self.test_file_count:, [4]] = 'test'
                    fold_df.to_csv(
                        self.__get_train_dev_test_k_fold_partition_df_path_2(reference_cdp, k)
                    )

                    progress_bar.update(1)

            print()

    def get_train_dev_test_k_fold_partition_df_2(self, 
                                                 reference_cdp: CodealltagDataProcessor,
                                                 k: int) -> DataFrame:
        return pd.read_csv(
            self.__get_train_dev_test_k_fold_partition_df_path_2(reference_cdp, k),
            index_col=0
        )
    
    
    def get_selected_seq2seq_pseudo_df(self, 
                                       reference_cdp: CodealltagDataProcessor,
                                       force_create: bool = False) -> DataFrame:
        
        selected_seq2seq_pseudo_df_path = self.__get_selected_seq2seq_pseudo_df_path(reference_cdp)
        if not os.path.exists(selected_seq2seq_pseudo_df_path) or force_create:
            self.__create_selected_seq2seq_pseudo_df(reference_cdp)
        return pd.read_csv(selected_seq2seq_pseudo_df_path, index_col=0)
    
    def get_selected_seq2seq_pseudo_datasetdict(self,
                                                reference_cdp: CodealltagDataProcessor,
                                                k: int = 1) -> DatasetDict:
        
        self.prepare_train_dev_test_k_folds_data_partitions_2(reference_cdp)
        seq2seq_pseudo_df = self.get_selected_seq2seq_pseudo_df(reference_cdp)
        kth_partition_df = self.get_train_dev_test_k_fold_partition_df_2(reference_cdp, k)
        
        df = kth_partition_df.merge(seq2seq_pseudo_df, how='inner', on='ID')
        
        train_ds = Dataset.from_pandas(df[df.Group=='train'])
        dev_ds = Dataset.from_pandas(df[df.Group=='dev'])
        test_ds = Dataset.from_pandas(df[df.Group=='test'])

        return DatasetDict({
            'train': train_ds,
            'dev': dev_ds,
            'test': test_ds
        })
    
        
    
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

    def __prepare_selection_tag(self) -> str:
        return '_'.join([
            str(self.random_seed),
            str(self.max_file_count),
            str(self.max_file_size),
            str(self.multiplier)
        ])

    def __get_selected_category_path_df_path(self) -> str:
        dir_path = os.path.join(self.data_dir_path, self.__get_xl_dir_prefix() + 'selected_' + self.data_version)
        os.makedirs(dir_path, exist_ok=True)
        file_name = self.__prepare_selection_tag() + '.csv'
        return os.path.join(dir_path, file_name)
    
    def __get_selected_seq2seq_df_path(self) -> str:
        dir_path = os.path.join(self.data_dir_path, self.__get_xl_dir_prefix() + 'selected_' + self.data_version)
        os.makedirs(dir_path, exist_ok=True)
        file_name = self.__prepare_selection_tag() + '_seq2seq' + '.csv'
        return os.path.join(dir_path, file_name)
    
    def __get_same_annotation_count_path_df_path(self, reference_cdp: CodealltagDataProcessor) -> str:
        file_name = 'SameAnnotationCountPath_DF_' + self.get_data_version() + '_'  + reference_cdp.get_data_version() + '.csv'
        return os.path.join(self.get_xl_dir_path(), file_name)
    
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
    
    def __create_selected_file_path_df(self) -> None:
        df = self.select_category_path_df(self.max_file_count, self.max_file_size, self.multiplier)
        df.to_csv(self.__get_selected_category_path_df_path())
    
    def __create_selected_seq2seq_df(self) -> None:
        selected_category_path_df = self.get_selected_category_path_df()
        selected_category_path_df_rows = selected_category_path_df.itertuples(index=True)
        seq2seq_df_row_tuples = list()
        with tqdm(total=len(selected_category_path_df), position=0, leave=True) as progress_bar:
            for row in selected_category_path_df_rows:
                seq_in = " ".join(self.get_somajo_split_sentences(row.FilePath))
                annotation_df = self.get_annotation_df_by_file(row.FilePath)
                seq_out = (annotation_df.Label + ": " + annotation_df.Token).str.cat(sep="; ")
                seq2seq_df_row_tuples.append((
                    row.ID,
                    seq_in,
                    seq_out
                ))    
                progress_bar.update(1)

        df = pd.DataFrame(seq2seq_df_row_tuples, columns=self.__get_selected_seq2seq_df_columns())
        df.to_csv(self.__get_selected_seq2seq_df_path())
    
    def __create_same_annotation_count_path_df(self, reference_cdp: CodealltagDataProcessor) -> None:
        category_path_df = self.get_category_path_df()
        category_path_df_rows = category_path_df.itertuples(index=True)
        total_files_with_same_annotation_count = 0
        files_with_same_annotation_count = list()
        with tqdm(total=len(category_path_df), position=0, leave=True) as progress_bar:
            for row in category_path_df_rows:
                annotation_df = self.get_annotation_df_by_file(row.FilePath)
                if os.path.exists(reference_cdp.get_absolute_annotations_file_path(row.FilePath)):
                    annotation_df_reference = reference_cdp.get_annotation_df_by_file(row.FilePath)
                    if len(annotation_df) == len(annotation_df_reference):
                        total_files_with_same_annotation_count += 1
                        files_with_same_annotation_count.append(row.FilePath)
                progress_bar.update(1)
        
        df = pd.DataFrame({'FilePath': files_with_same_annotation_count})
        df.to_csv(self.__get_same_annotation_count_path_df_path())
    
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
        df.to_csv(self.__get_same_entity_labels_path_df_path())
    
    def __prepare_selection_tag_2(self, reference_cdp: CodealltagDataProcessor) -> str:
        return '_'.join([
            str(reference_cdp.get_data_version()),
            str(self.random_seed),
            str(self.max_file_count),
            str(self.max_file_size),
            str(self.multiplier)
        ])
    
    def __get_selected_same_annotation_count_path_df_path(self, reference_cdp: CodealltagDataProcessor) -> str:
        dir_path = os.path.join(self.data_dir_path, self.__get_xl_dir_prefix() + 'selected_' + self.data_version)
        os.makedirs(dir_path, exist_ok=True)
        file_name = self.__prepare_selection_tag_2(reference_cdp) + '.csv'
        return os.path.join(dir_path, file_name)
    
    def __create_selected_same_annotation_count_path_df(self, reference_cdp: CodealltagDataProcessor) -> None:
        df = self.select_same_annotation_count_path_df(
            reference_cdp,
            self.max_file_count,
            self.max_file_size,
            self.multiplier
        )
        df.to_csv(self.__get_selected_same_annotation_count_path_df_path(reference_cdp))
    
    def __get_selected_seq2seq_pseudo_df_path(self, reference_cdp: CodealltagDataProcessor) -> str:
        dir_path = os.path.join(self.data_dir_path, self.__get_xl_dir_prefix() + 'selected_' + self.data_version)
        os.makedirs(dir_path, exist_ok=True)
        file_name = self.__prepare_selection_tag_2(reference_cdp) + '_seq2seq_pseudo' + '.csv'
        return os.path.join(dir_path, file_name)
    
    def __create_selected_seq2seq_pseudo_df(self, reference_cdp: CodealltagDataProcessor) -> None:
        selected_same_annotation_count_path_df = self.get_selected_same_annotation_count_path_df(reference_cdp)
        rows = selected_same_annotation_count_path_df.itertuples(index=True)
        seq2seq_pseudo_df_row_tuples = list()
        with tqdm(total=len(selected_same_annotation_count_path_df), position=0, leave=True) as progress_bar:
            for row in rows:
                seq_in = " ".join(self.get_somajo_split_sentences(row.FilePath))
                annotation_df = self.get_annotation_df_by_file(row.FilePath)
                annotation_df_reference = reference_cdp.get_annotation_df_by_file(row.FilePath)
                seq_out = (
                    annotation_df.Label +
                    ": " +
                    annotation_df.Token + 
                    " **" + 
                    annotation_df_reference.Token +
                    "**"
                ).str.cat(sep="; ")
                seq2seq_pseudo_df_row_tuples.append((
                    row.ID,
                    seq_in,
                    seq_out
                ))    
                progress_bar.update(1)

        df = pd.DataFrame(seq2seq_pseudo_df_row_tuples, columns=self.__get_selected_seq2seq_df_columns())
        df.to_csv(self.__get_selected_seq2seq_pseudo_df_path(reference_cdp))
    
    
    def __get_processed_dir_path(self) -> str:
        return os.path.join(
            self.data_dir_path,
            self.__get_xl_dir_prefix() + 'processed_' + self.data_version,
            self.tokenizer_version + '_' + self.__prepare_selection_tag()
        )

    def __get_partitions_dir_path(self) -> str:
        return os.path.join(
            self.__get_processed_dir_path(),
            'partitions_' + str(self.test_file_count) + '_k' + str(self.k_fold)
        )

    def __get_train_dev_test_k_fold_dir_path(self, k: int) -> str:
        return os.path.join(self.__get_partitions_dir_path(), 'k' + str(k))

    def __get_train_dev_test_k_fold_partition_df_path(self, k: int) -> str:
        return os.path.join(self.__get_train_dev_test_k_fold_dir_path(k), 'partition.csv')
    
    def __get_processed_dir_path_2(self, reference_cdp: CodealltagDataProcessor) -> str:
        return os.path.join(
            self.data_dir_path,
            self.__get_xl_dir_prefix() + 'processed_' + self.data_version,
            self.__prepare_selection_tag_2(reference_cdp)
        )
    
    def __get_partitions_dir_path_2(self, reference_cdp: CodealltagDataProcessor) -> str:
        return os.path.join(
            self.__get_processed_dir_path_2(reference_cdp),
            'partitions_' + str(self.test_file_count) + '_k' + str(self.k_fold)
        )
    
    def __get_train_dev_test_k_fold_dir_path_2(self, reference_cdp: CodealltagDataProcessor, k: int) -> str:
        return os.path.join(self.__get_partitions_dir_path_2(reference_cdp), 'k' + str(k))

    def __get_train_dev_test_k_fold_partition_df_path_2(self, reference_cdp: CodealltagDataProcessor, k: int) -> str:
        return os.path.join(self.__get_train_dev_test_k_fold_dir_path_2(reference_cdp, k), 'partition.csv')
    
    
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
    def __get_config_key_tokenizer_version() -> str:
        return 'tokenizer_version'

    @staticmethod
    def __get_config_key_max_file_count() -> str:
        return 'max_file_count'

    @staticmethod
    def __get_config_key_max_file_size() -> str:
        return 'max_file_size'

    @staticmethod
    def __get_config_key_multiplier() -> str:
        return 'multiplier'
    
    @staticmethod
    def __get_config_key_test_file_count() -> str:
        return 'test_file_count'

    @staticmethod
    def __get_config_key_k_fold() -> str:
        return 'k_fold'

    @staticmethod
    def __get_config_key_use_fold() -> str:
        return 'use_fold'

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
    def __get_selected_category_path_df_columns() -> List[str]:
        return ['ID', 'Category', 'FilePath', 'FileSize', 'Group']
    
    @staticmethod
    def __get_selected_seq2seq_df_columns() -> List[str]:
        return ['ID', 'SeqIn', 'SeqOut']
    
    @staticmethod
    def __get_somajo_tokenizer() -> SoMaJo:
        return SoMaJo("de_CMC", split_camel_case=False)


    # property

    config_path_str = property(get_config_path_str)
    random_seed = property(get_random_seed)
    data_dir_path = property(get_data_dir_path)
    data_version = property(get_data_version)
    tokenizer_version = property(get_tokenizer_version)
    max_file_count = property(get_max_file_count)
    max_file_size = property(get_max_file_size)
    multiplier = property(get_multiplier)
    test_file_count = property(get_test_file_count)
    k_fold = property(get_k_fold)
    use_fold = property(get_use_fold)
    xl_dir_path = property(get_xl_dir_path)
    xl_ann_dir_path = property(get_xl_ann_dir_path)