"""Locked dataset protocols for controlled comparison experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ProtocolSpec:
    dataset: str
    protocol_version: str
    selection_train_list: str
    full_train_list: str
    val_gallery_list: str
    val_protocol_list: str
    test_gallery_list: str
    test_protocol_list: str
    palm_checkpoint: str
    vein_checkpoint: str
    selection_train_sha256: str
    full_train_sha256: str
    val_gallery_sha256: str
    val_protocol_sha256: str
    test_gallery_sha256: str
    test_protocol_sha256: str
    palm_checkpoint_sha256: str
    vein_checkpoint_sha256: str
    val_gallery_identities: int
    val_probes_per_scenario: int
    test_gallery_identities: int
    test_probes_per_scenario: int
    selection_num_classes: int
    full_num_classes: int
    validation_split_path: str | None = None
    validation_split_sha256: str | None = None

    @property
    def selection_fingerprints(self) -> dict[str, str]:
        return self.training_fingerprints(self.selection_train_sha256)

    @property
    def full_fingerprints(self) -> dict[str, str]:
        return self.training_fingerprints(self.full_train_sha256)

    @property
    def test_fingerprints(self) -> dict[str, str]:
        return {
            "gallery_protocol_sha256": self.test_gallery_sha256,
            "probe_protocol_sha256": self.test_protocol_sha256,
        }

    def training_fingerprints(self, train_sha256: str) -> dict[str, str]:
        return {
            "palm_encoder_sha256": self.palm_checkpoint_sha256,
            "vein_encoder_sha256": self.vein_checkpoint_sha256,
            "train_list_sha256": train_sha256,
            "validation_gallery_sha256": self.val_gallery_sha256,
            "validation_probe_sha256": self.val_protocol_sha256,
        }


PROTOCOL_SPECS = {
    "tongji": ProtocolSpec(
        dataset="tongji",
        protocol_version="tongji_identity_validation_v1",
        selection_train_list="data_txt/tongji/ssfd_train_full.txt",
        full_train_list="data_txt/tongji/ssfd_train_full.txt",
        val_gallery_list="data_txt/tongji/ssfd_val_gallery_full.txt",
        val_protocol_list="data_txt/tongji/ssfd_val_protocol.txt",
        test_gallery_list="data_txt/tongji/ssfd_gallery_full.txt",
        test_protocol_list="data_txt/tongji/ssfd_test_protocol.txt",
        palm_checkpoint="outputs/encoders/palm_best.pth",
        vein_checkpoint="outputs/encoders/vein_best.pth",
        selection_train_sha256="223a3ba6a65633a71a79779b169ca32c2009240eea0ac131bf48b7ff66a2466d",
        full_train_sha256="223a3ba6a65633a71a79779b169ca32c2009240eea0ac131bf48b7ff66a2466d",
        val_gallery_sha256="a78401f1ae1a0e7bbbe507c473782d0bd3f6873a907c6eb2f83799e84ec98d2f",
        val_protocol_sha256="cab788585a32e8803f137d4ff20260578097eaec1685d50604de853b15aeb55a",
        test_gallery_sha256="a4d3b3cd4d5765bb9c084adb09e6bd211716de9e44a8bf5ac05685e5ac9af2bd",
        test_protocol_sha256="e5484ab82b23a4f609714d24e94f4916a40655475258cd37b3dc4abf560d29a3",
        palm_checkpoint_sha256="44fd9ac987dfc11e8bf40f37419f437a5bc611729e4a7ad71835eaed6a64438e",
        vein_checkpoint_sha256="2925ee5d89962ad6d9dafa97a047f1c4a9d2675e11487114fb0f7230bdcaf122",
        val_gallery_identities=48,
        val_probes_per_scenario=96,
        test_gallery_identities=120,
        test_probes_per_scenario=240,
        selection_num_classes=432,
        full_num_classes=432,
    ),
    "cumt": ProtocolSpec(
        dataset="cumt",
        protocol_version="cumt_identity_8_2_recovery_val_v1",
        selection_train_list="data_txt/cumt/ssfd_recovery_train.txt",
        full_train_list="data_txt/cumt/ssfd_train_full.txt",
        val_gallery_list="data_txt/cumt/ssfd_recovery_val_gallery.txt",
        val_protocol_list="data_txt/cumt/ssfd_recovery_val_protocol.txt",
        test_gallery_list="data_txt/cumt/ssfd_gallery_full.txt",
        test_protocol_list="data_txt/cumt/ssfd_test_protocol.txt",
        palm_checkpoint="outputs/encoders/identity_8_2/cumt/palm_best.pth",
        vein_checkpoint="outputs/encoders/identity_8_2/cumt/vein_best.pth",
        selection_train_sha256="d8096d0698a5d6610a9c2eb7bc044aee0f6eaa63b6fe84adff0bdc44eeeec9ee",
        full_train_sha256="d67447b163d6629d421b45b955d10d7ca00544724dac8b7ca2381a3dbb374390",
        val_gallery_sha256="d9a9d304bceb7a7e19bc0efa5d8f732ab1fd817d6dc8a9160bd59c4c0a85158a",
        val_protocol_sha256="a4076e348811dd345d3d481c4864bc08c5f7686c69a4f78ac319fcf22598aece",
        test_gallery_sha256="923b95541d2dce629261bdfb44954730f7cd2825ba817425d652c5e5da7486bb",
        test_protocol_sha256="efc9d1f37a3236ea3140f14b219f6e64b92ab7b86a8ea69ec7d08931753f5d3d",
        palm_checkpoint_sha256="e7d39aa5e615ed22d0a2662ac902b12c697cb626d457405e83790767ce1e4cd0",
        vein_checkpoint_sha256="246cdf579d46e359d1eb2fffd8484a6b20e7bc4731f10f6cbfacb1f357ece83a",
        val_gallery_identities=23,
        val_probes_per_scenario=46,
        test_gallery_identities=58,
        test_probes_per_scenario=116,
        selection_num_classes=209,
        full_num_classes=232,
        validation_split_path="data_txt/cumt/recovery_validation_split.json",
        validation_split_sha256="3d15da0f370447bc8afb5feb4f1651383aaf8694370dbc7523421d4207a3f7a5",
    ),
    "polyu": ProtocolSpec(
        dataset="polyu",
        protocol_version="polyu_identity_8_2_recovery_val_v1",
        selection_train_list="data_txt/polyu/ssfd_recovery_train.txt",
        full_train_list="data_txt/polyu/ssfd_train_full.txt",
        val_gallery_list="data_txt/polyu/ssfd_recovery_val_gallery.txt",
        val_protocol_list="data_txt/polyu/ssfd_recovery_val_protocol.txt",
        test_gallery_list="data_txt/polyu/ssfd_gallery_full.txt",
        test_protocol_list="data_txt/polyu/ssfd_test_protocol.txt",
        palm_checkpoint="outputs/encoders/identity_8_2/polyu/palm_best.pth",
        vein_checkpoint="outputs/encoders/identity_8_2/polyu/vein_best.pth",
        selection_train_sha256="d08e591b1d3e2cbc8e76aa8fe58c67929e779de9d0532585cc392c8bc2668d83",
        full_train_sha256="00d0f4e0776c09a7eae3bf69d3bb1d9d86b3c9e41b888684de295480859c5cca",
        val_gallery_sha256="a1826e0382841bb295ed2ec617f5165258a1a10154b201b77763555ff4774566",
        val_protocol_sha256="81334b4fbcd8c07a390712079f99f947cb16e1a85c38e057d2a9cf48c613ec39",
        test_gallery_sha256="b0347cafeac1f7d34a77f061da51c6eb30df0d7b4f941b3a34863c99e5b7ebf5",
        test_protocol_sha256="0c8ead06cd4820384e1545f6032f1b46f822e06b0ba1fab2f94424525528987f",
        palm_checkpoint_sha256="22348f02d6277a9a1761bb5e299f65a82dbdaf50aaf663c94a88485ae1ee977d",
        vein_checkpoint_sha256="4d8874fee28ba0ab76e78b907b9032081a4a579a1b60752562c43579fb981e94",
        val_gallery_identities=40,
        val_probes_per_scenario=80,
        test_gallery_identities=100,
        test_probes_per_scenario=200,
        selection_num_classes=360,
        full_num_classes=400,
        validation_split_path="data_txt/polyu/recovery_validation_split.json",
        validation_split_sha256="2a92793dbc7d79514df07e05be751fcfced37964521ef298cbb41628d29f05ab",
    ),
}

DATASETS = tuple(PROTOCOL_SPECS)


def get_protocol_spec(dataset: str) -> ProtocolSpec:
    try:
        return PROTOCOL_SPECS[dataset.lower()]
    except (AttributeError, KeyError) as error:
        raise ValueError(
            f"Unsupported comparison dataset {dataset!r}; choose one of {DATASETS}"
        ) from error
