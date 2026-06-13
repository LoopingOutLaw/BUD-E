"""Tests for the text encoder + BPE tokenizer."""
import torch
from bude_vla.models.text_encoder import SimpleTokenizer, TinyTextEncoder, PAD_ID, BOS_ID


def test_tokenizer_train_encode_decode():
    tok = SimpleTokenizer(vocab_size=128, max_len=16)
    corpus = [
        "pick red cube", "pick blue cube", "reach target",
        "push cube to blue zone", "drop in bowl",
    ] * 5  # repeat to give BPE enough signal
    tok.train(corpus)
    ids = tok.encode("pick red cube")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    assert ids[0] == BOS_ID
    decoded = tok.decode(ids)
    assert "pick" in decoded


def test_tokenizer_pad_BOS_EOS():
    tok = SimpleTokenizer(vocab_size=64, max_len=8)
    tok.train(["hello world"] * 3)
    ids = tok.encode("hello world")
    # Should be padded to max_len
    assert len(ids) == 8
    assert ids[0] == BOS_ID


def test_tokenizer_batch_encode_shape():
    tok = SimpleTokenizer(vocab_size=64, max_len=8)
    tok.train(["foo bar", "baz qux", "lorem ipsum"] * 3)
    out = tok.batch_encode(["foo bar", "baz qux"])
    assert out.shape == (2, 8)
    assert out.dtype == torch.long


def test_tokenizer_save_load(tmp_path):
    tok = SimpleTokenizer(vocab_size=32, max_len=8)
    tok.train(["hello world"] * 3)
    p = tmp_path / "tok.json"
    tok.save(p)
    tok2 = SimpleTokenizer(vocab_size=32, max_len=8)
    tok2.load(p)
    a = tok.encode("hello world")
    b = tok2.encode("hello world")
    assert a == b


def test_text_encoder_output_shape():
    e = TinyTextEncoder(vocab_size=64, max_len=12, d=32, depth=2, heads=2)
    ids = torch.randint(0, 64, (3, 12))
    out = e(ids)
    assert out.shape == (3, 12, 32)


def test_text_encoder_masking_real_pad():
    """Tokenizer-encoded texts (with real PAD positions) should still produce outputs."""
    tok = SimpleTokenizer(vocab_size=64, max_len=8)
    tok.train(["hello world test"] * 3)
    ids = tok.batch_encode(["hello", "hello long sentence here hopefully"])
    e = TinyTextEncoder(vocab_size=64, max_len=8, d=32, depth=2, heads=2)
    out = e(ids)
    assert out.shape == (2, 8, 32)
    # Output at PAD positions should generally differ from BOS due to noised info,
    # but at minimum it shouldn't NaN.
    assert torch.isfinite(out).all()
