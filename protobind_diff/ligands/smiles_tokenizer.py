# Taken from https://github.com/MolecularAI/Chemformer/
from typing import Any, Dict, List, Optional, Tuple, Union
from pysmilesutils.tokenize import SMILESTokenizer

class ChemformerTokenizer(SMILESTokenizer):
    """
    Tokenizer for the Chemformer.

    There are a few different features that sets this apart from the `SMILESTokenizer`:
       * It reserves two extra special tokens, "mask" and "sep"
       * It distinguish between chemical and non-chemical tokens

    :param smiles: A list of SMILES that are used to create the vocabulary for the tokenizer. Defaults to None.
    :param tokens:  A list of tokens (strings) that the tokenizer uses when tokenizing SMILES. Defaults to None.
    :param regex_token_patterns: A list of regular expressions that the tokenizer uses when tokenizing SMILES.
    :param beginning_of_smiles_token: Token that is added to beginning of SMILES. Defaults to "^".
    :param end_of_smiles_token: Token that is added to the end of SMILES. Defaults to "&".
    :param padding_token: Token used for padding. Defalts to " ".
    :param unknown_token: Token that is used for unknown ids when decoding encoded data. Defaults to "?".
    :param mask_token: Token that is used by the Masker
    :param sep_token: Token that is used to separate sentences, currently unused
    :param filename: if given and `smiles` is None, load the vocabulary from disc
    :raises: ValueError: If the `encoding_type` is invalid.
    """

    def __init__(
        self,
        smiles: List[str] = None,
        tokens: List[str] = None,
        regex_token_patterns: List[str] = None,
        beginning_of_smiles_token: str = "^",
        end_of_smiles_token: str = "&",
        padding_token: str = "<PAD>",
        unknown_token: str = "?",
        mask_token: str = "<MASK>",
        sep_token: str = "<SEP>",
        filename: str = None,
    ) -> None:
        self._mask_token = mask_token
        self._sep_token = sep_token
        self._chem_start_idx = 6  # Default, number of special tokens + 1
        self._chem_token_idxs: Optional[List[int]] = None
        super().__init__(
            smiles=smiles,
            tokens=tokens,
            regex_token_patterns=regex_token_patterns,
            beginning_of_smiles_token=beginning_of_smiles_token,
            end_of_smiles_token=end_of_smiles_token,
            padding_token=padding_token,
            unknown_token=unknown_token,
            encoding_type="index",
            filename=filename,
        )


    @property
    def chem_token_idxs(self) -> List[int]:
        """Returns the indices of the vocabulary that are chemical tokens"""
        if self._chem_token_idxs is None:
            self._chem_token_idxs = list(range(self._chem_start_idx, len(self.vocabulary)))
        return self._chem_token_idxs

    @property
    def mask_token_id(self):
        """Get the mask token id"""
        return self.vocabulary[self._mask_token]
    
    @property
    def vocab_size(self):
        return len(self.vocabulary)

    @property
    def special_tokens(self) -> Dict[str, str]:
        """Returns a dictionary of non-character tokens"""
        return {
            "start": self._beginning_of_smiles_token,
            "end": self._end_of_smiles_token,
            "pad": self._padding_token,
            "unknown": self._unknown_token,
            "mask": self._mask_token,
            "sep": self._sep_token,
        }

    def add_tokens(self, tokens: List[str], regex: bool = False, smiles=None) -> None:
        """Adds tokens to the classes list of tokens.

        The new tokens are added to the front of the token list and take priority over old tokens. Note that that the
        vocabulary of the tokenizer is not updated after the tokens are added,
        and must be updated by calling `create_vocabulary_from_smiles`.

        If `regex` is False, the tokens are interpreted as non-chemical tokens, which distinguish
        them for processing by e.g. the masker.

        :param tokens: List of tokens to be added.
        :param regex: If `True` the input tokens are treated as
                regular expressions and are added to the list of regular expressions
                instead of token list. Defaults to False.
        :param smiles: If a list of smiles is provided, the vocabulary will be created, defaults to None

        :raises ValueError: If any of the tokens supplied are already in the list
                of tokens.
        """
        super().add_tokens(tokens, regex, smiles)
        if not regex:
            self._chem_start_idx += len(tokens)
            self._chem_token_idxs = None

    def _reset_vocabulary(self) -> Dict[str, int]:
        """Create a new tokens vocabulary.

        :return: New tokens vocabulary
        """
        dict_ = {
            self._padding_token: 0,
            self._unknown_token: 1,
            self._beginning_of_smiles_token: 2,
            self._end_of_smiles_token: 3,
            self._mask_token: 4,
            self._sep_token: 5,
        }
        for token in self._tokens:
            dict_.setdefault(token, len(dict_))
        return dict_

    def _state_properties(self) -> Dict[str, Any]:
        """Return properties to reconstruct the internal state of the tokenizer"""
        dict_ = super()._state_properties()
        dict_["chem_start_idx"] = self._chem_start_idx
        return dict_

    def _update_state(self, dict_: Dict[str, Any]) -> None:
        """Update the internal state with properties loaded from disc"""
        super()._update_state(dict_)
        self._chem_start_idx = dict_["chem_start_idx"]
        self._chem_token_idxs = None