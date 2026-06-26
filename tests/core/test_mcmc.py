import inspect
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from src.bffg import (
    MCMCModelConfig,
    MCMCParams,
    build_bffg_context,
    covariance_matrix,
    diffusion_matrix,
    dot_factorized,
    forward_guided,
    inverse_gamma_logpdf,
    log_gaussian_prec,
    mcmc_log_posterior,
    mcmc_log_posterior_and_node_state,
    normal_logpdf,
    solve_factorized,
)
from src.tree_state import BFFG_FIELDS, bffg_schema, zero_bffg_arrays
from src.driver import (
    MCMCDriverConfig,
    initial_params_for_chain,
    propose_params,
    run_mcmc_chain,
    run_mcmc_chains,
    run_mcmc,
    sample_initial_params,
)
from src.loader import load_augmented_butterfly_tree
from src.kunita import covariance_matrix as kunita_covariance_matrix
from src.kunita import diffusion_matrix as kunita_diffusion_matrix
from src.kunita import kernel_matrix as kunita_kernel_matrix
from src.kunita import laplace_k1_kernel
from src.run_artifacts import prepare_run_dir


REPO_ROOT = Path(__file__).resolve().parents[2]
BUTTERFLIES_H5 = REPO_ROOT / "data/butterflies/data.h5"


def test_kernel_uses_current_laplace_covariance_and_cholesky():
    q = jnp.array(
        [
            [-0.4, 0.1],
            [0.3, -0.2],
            [0.5, 0.4],
        ],
        dtype=jnp.float32,
    )
    params = MCMCParams(k_alpha=0.2, k_sigma=0.25, obs_var=1e-3)
    pair_diff = q[:, None, :] - q[None, :, :]

    kernel = kunita_kernel_matrix(q, params, d_landmarks=2, dist_jitter=1e-7)
    sigma = diffusion_matrix(q, params, covar_jitter=1e-8, dist_jitter=1e-7)
    cov = covariance_matrix(q, params, covar_jitter=1e-8, dist_jitter=1e-7)

    np.testing.assert_allclose(
        np.asarray(kernel),
        np.asarray(laplace_k1_kernel(pair_diff, params, dist_jitter=1e-7)),
    )
    np.testing.assert_allclose(np.asarray(cov), np.asarray(kernel + 1e-8 * jnp.eye(3)), rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(np.asarray(sigma), np.tril(np.asarray(sigma)))
    np.testing.assert_allclose(np.asarray(cov), np.asarray(sigma @ sigma.T), rtol=1e-5, atol=1e-7)
    assert cov.shape == (3, 3)


def test_kunita_diffusion_module_matches_bffg_reexports():
    q = jnp.array([[-0.4, 0.1], [0.3, -0.2], [0.5, 0.4]], dtype=jnp.float32)
    params = MCMCParams(k_alpha=0.2, k_sigma=0.25, obs_var=1e-3)

    np.testing.assert_allclose(
        np.asarray(kunita_covariance_matrix(q, params, covar_jitter=1e-8, dist_jitter=1e-7)),
        np.asarray(covariance_matrix(q, params, covar_jitter=1e-8, dist_jitter=1e-7)),
    )
    np.testing.assert_allclose(
        np.asarray(kunita_diffusion_matrix(q, params, covar_jitter=1e-8, dist_jitter=1e-7)),
        np.asarray(diffusion_matrix(q, params, covar_jitter=1e-8, dist_jitter=1e-7)),
    )


def test_kunita_covariance_accepts_flat_3d_landmark_state():
    q = jnp.array([0.0, 0.0, 0.0, 0.0, 3.0, 4.0], dtype=jnp.float32)
    params = MCMCParams(k_alpha=0.2, k_sigma=0.25, obs_var=1e-3)

    cov = kunita_covariance_matrix(
        q,
        params,
        d_landmarks=3,
        covar_jitter=1e-8,
        dist_jitter=1e-7,
    )
    expected_kernel = kunita_kernel_matrix(
        q.reshape((2, 3)),
        params,
        d_landmarks=3,
        dist_jitter=1e-7,
    )

    assert cov.shape == (2, 2)
    np.testing.assert_allclose(
        np.asarray(cov),
        np.asarray(expected_kernel + 1e-8 * jnp.eye(2)),
        rtol=1e-6,
        atol=1e-8,
    )


def test_active_src_mcmc_does_not_import_external_bffg_or_numpyro_driver():
    import src.bffg as core
    import src.driver as driver

    core_source = inspect.getsource(core)
    driver_source = inspect.getsource(driver)

    external_bffg_module = "hyperiax." + "pre" + "built.bffg"
    assert external_bffg_module not in core_source
    assert "continuous_bf_sweep" not in core_source
    assert "jnp.kron" not in core_source
    assert "numpyro" not in driver_source
    assert "MCMC(" not in driver_source


def test_bffg_message_state_schema_centralizes_factorized_tree_fields():
    schema = bffg_schema(num_edge_steps=2, n_landmarks=118, d_landmarks=2)
    zeros = zero_bffg_arrays(
        node_count=34,
        num_edge_steps=2,
        n_landmarks=118,
        state_dim=236,
        dtype=jnp.float32,
    )

    assert BFFG_FIELDS.edge_len == "edge_len"
    assert BFFG_FIELDS.prec_v == "prec_v"
    assert BFFG_FIELDS.tildea_v == "tildea_v"
    assert schema[BFFG_FIELDS.vals] == (3, 236)
    assert schema[BFFG_FIELDS.precs] == (3, 118, 118)
    assert zeros[BFFG_FIELDS.zs].shape == (34, 2, 236)
    assert zeros[BFFG_FIELDS.prec_v].shape == (34, 118, 118)


def test_bffg_context_uses_hdf5_contract_and_factorized_fields():
    dataset = load_augmented_butterfly_tree(BUTTERFLIES_H5)
    context = build_bffg_context(dataset, MCMCModelConfig(num_edge_steps=2))

    assert context.node_count == 850
    assert context.n_landmarks == 118
    assert context.d_landmarks == 2
    assert context.state_dim == 236
    assert context.noise_size == 850 * 2 * 236
    assert context.leaf_observations.shape == (425, 236)
    assert context.tree[BFFG_FIELDS.vals].shape == (850, 3, 236)
    assert context.tree[BFFG_FIELDS.zs].shape == (850, 2, 236)
    assert context.tree[BFFG_FIELDS.precs].shape == (850, 3, 118, 118)
    assert context.tree[BFFG_FIELDS.prec_v].shape == (850, 118, 118)
    assert np.isfinite(np.asarray(context.root_value)).all()
    assert np.isfinite(np.asarray(context.leaf_observations)).all()
    assert int(np.asarray(dataset.tree.topology.is_leaf).sum()) == 425
    np.testing.assert_allclose(
        np.asarray(context.tree[BFFG_FIELDS.edge_len]),
        dataset.edge_len_norm,
    )


def test_bffg_context_always_uses_normalized_hdf5_edge_lengths():
    dataset = load_augmented_butterfly_tree(BUTTERFLIES_H5)

    context = build_bffg_context(dataset, MCMCModelConfig(num_edge_steps=1))

    np.testing.assert_allclose(np.asarray(context.tree[BFFG_FIELDS.edge_len]), dataset.edge_len_norm, rtol=1e-6)
    assert np.isclose(context.edge_len_normalizer, dataset.edge_len_normalizer)
    assert not np.allclose(
        np.asarray(context.tree[BFFG_FIELDS.edge_len]),
        dataset.edge_len_aug,
    )


def test_mcmc_sweeps_match_direct_tiny_reference_for_one_target():
    dataset = _tiny_augmented_dataset()
    config = MCMCModelConfig(num_edge_steps=1)
    context = build_bffg_context(dataset, config)
    params = MCMCParams(k_alpha=0.18, k_sigma=0.35, obs_var=0.02)
    noise = jnp.zeros_like(context.tree[BFFG_FIELDS.zs])

    swept = mcmc_log_posterior(context.tree, context.leaf_observations, params, noise, config)
    direct = _direct_tiny_log_posterior(context.root_value, context.leaf_observations, params, noise, config)

    np.testing.assert_allclose(float(swept), float(direct), rtol=1e-5, atol=1e-5)


def test_mcmc_target_and_phylo_root_samples_accept_3d_landmarks():
    dataset = _tiny_augmented_dataset_3d_with_phylo_root()
    config = MCMCModelConfig(num_edge_steps=1, covar_jitter=1e-6, dist_jitter=1e-7)
    context = build_bffg_context(dataset, config)
    params = MCMCParams(k_alpha=0.18, k_sigma=0.35, obs_var=0.02)
    noise = jnp.zeros_like(context.tree[BFFG_FIELDS.zs])

    logp, phylo_root_state = mcmc_log_posterior_and_node_state(
        context.tree,
        context.leaf_observations,
        params,
        noise,
        config,
        node_index=1,
    )
    result = run_mcmc_chain(
        context,
        driver_config=MCMCDriverConfig(num_samples=1, num_chains=1, progress_bar=False),
        rng_key=jax.random.PRNGKey(42),
        initial_state=(params, noise),
        collect_phylo_root_samples=True,
    )

    assert context.n_landmarks == 2
    assert context.d_landmarks == 3
    assert np.isfinite(float(logp))
    assert phylo_root_state.shape == (6,)
    assert result.phylo_root_samples.shape == (1, 2, 3)


def test_run_mcmc_chain_smoke_samples_obs_var():
    dataset = _tiny_augmented_dataset()
    context = build_bffg_context(dataset, MCMCModelConfig(num_edge_steps=1))
    driver_config = MCMCDriverConfig(
        num_samples=4,
        num_chains=1,
        random_seed=12,
        progress_bar=False,
    )

    result = run_mcmc_chain(context, driver_config=driver_config, rng_key=jax.random.PRNGKey(12))

    assert result.log_posteriors.shape == (4,)
    assert result.samples["k_alpha"].shape == (4,)
    assert result.samples["k_sigma"].shape == (4,)
    assert result.samples["obs_var"].shape == (4,)
    assert result.accepted.shape == (4,)
    assert 0.0 <= result.acceptance_rate <= 1.0
    assert np.isfinite(result.log_posteriors).all()
    assert float(result.initial_params.obs_var) > 0.0
    assert np.isfinite(result.samples["obs_var"]).all()


def test_run_mcmc_chain_clamps_explicit_initial_state_to_model_bounds():
    dataset = _tiny_augmented_dataset()
    context = build_bffg_context(
        dataset,
        MCMCModelConfig(
            num_edge_steps=1,
            k_alpha_min=0.01,
            k_alpha_max=0.1,
            k_sigma_min=0.2,
            k_sigma_max=0.5,
            obs_var_min=1e-4,
            obs_var_max=1e-2,
        ),
    )
    initial_noise = jnp.zeros_like(context.tree[BFFG_FIELDS.zs])
    driver_config = MCMCDriverConfig(
        num_samples=1,
        num_chains=1,
        progress_bar=False,
        k_alpha_proposal_var=0.0,
        k_sigma_proposal_var=0.0,
        obs_var_proposal_var=0.0,
    )

    result = run_mcmc_chain(
        context,
        driver_config=driver_config,
        rng_key=jax.random.PRNGKey(11),
        initial_state=(MCMCParams(k_alpha=0.001, k_sigma=10.0, obs_var=1e-8), initial_noise),
        target=lambda params, noise: jnp.asarray(0.0),
    )

    np.testing.assert_allclose(float(result.initial_params.k_alpha), 0.01)
    np.testing.assert_allclose(float(result.initial_params.k_sigma), 0.5)
    np.testing.assert_allclose(float(result.initial_params.obs_var), 1e-4)
    np.testing.assert_allclose(result.samples["k_alpha"], [0.01])
    np.testing.assert_allclose(result.samples["k_sigma"], [0.5])
    np.testing.assert_allclose(result.samples["obs_var"], [1e-4])


def test_run_mcmc_chain_progress_bar_reports_log_posterior_and_acceptance_rate(monkeypatch):
    class FakeTqdm:
        instances = []

        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            self.kwargs = kwargs
            self.postfixes = []
            FakeTqdm.instances.append(self)

        def __iter__(self):
            for item in self.iterable:
                yield item

        def set_postfix(self, ordered_dict=None, **kwargs):
            values = dict(ordered_dict or {})
            values.update(kwargs)
            self.postfixes.append(values)

    import src.driver as driver

    monkeypatch.setattr(driver, "tqdm", FakeTqdm)
    dataset = _tiny_augmented_dataset()
    context = build_bffg_context(dataset, MCMCModelConfig(num_edge_steps=1))
    driver_config = MCMCDriverConfig(
        num_samples=4,
        num_chains=1,
        random_seed=12,
        progress_bar=True,
    )

    result = run_mcmc_chain(context, driver_config=driver_config, rng_key=jax.random.PRNGKey(12))

    assert len(FakeTqdm.instances) == 1
    postfixes = FakeTqdm.instances[0].postfixes
    expected_acceptance_rates = np.cumsum(result.accepted) / np.arange(1, result.accepted.size + 1)
    assert [postfix["log_posterior"] for postfix in postfixes] == [
        f"{log_posterior:.3f}" for log_posterior in result.log_posteriors
    ]
    assert [postfix["accept_rate"] for postfix in postfixes] == [
        f"{acceptance_rate:.3f}" for acceptance_rate in expected_acceptance_rates
    ]


def test_initial_params_are_seeded_prior_draws():
    model_config = MCMCModelConfig(k_sigma_max=0.7)
    driver_config = MCMCDriverConfig()

    first = sample_initial_params(
        jax.random.PRNGKey(3),
        driver_config=driver_config,
        model_config=model_config,
    )
    repeat = sample_initial_params(
        jax.random.PRNGKey(3),
        driver_config=driver_config,
        model_config=model_config,
    )
    second = sample_initial_params(
        jax.random.PRNGKey(4),
        driver_config=driver_config,
        model_config=model_config,
    )

    np.testing.assert_allclose(float(first.k_alpha), float(repeat.k_alpha))
    np.testing.assert_allclose(float(first.k_sigma), float(repeat.k_sigma))
    np.testing.assert_allclose(float(first.obs_var), float(repeat.obs_var))
    assert float(first.k_alpha) > 0.0
    assert 0.0 < float(first.k_sigma) <= 0.7
    assert float(first.obs_var) > 0.0
    assert not np.allclose(
        [float(first.k_alpha), float(first.k_sigma), float(first.obs_var)],
        [float(second.k_alpha), float(second.k_sigma), float(second.obs_var)],
    )


def test_initial_params_are_clamped_to_model_parameter_bounds():
    model_config = MCMCModelConfig(
        k_alpha_init=0.001,
        k_alpha_min=0.01,
        k_alpha_max=0.1,
        k_sigma_init=10.0,
        k_sigma_min=0.2,
        k_sigma_max=0.5,
        obs_var_init=1e-8,
        obs_var_min=1e-4,
        obs_var_max=1e-2,
    )
    driver_config = MCMCDriverConfig(num_chains=1)

    params = initial_params_for_chain(
        jax.random.PRNGKey(7),
        chain_index=0,
        driver_config=driver_config,
        model_config=model_config,
    )

    np.testing.assert_allclose(float(params.k_alpha), 0.01)
    np.testing.assert_allclose(float(params.k_sigma), 0.5)
    np.testing.assert_allclose(float(params.obs_var), 1e-4)


def test_proposed_params_are_clamped_to_model_parameter_bounds():
    model_config = MCMCModelConfig(
        k_alpha_min=0.01,
        k_alpha_max=0.1,
        k_sigma_min=0.2,
        k_sigma_max=0.5,
        obs_var_min=1e-4,
        obs_var_max=1e-2,
    )
    driver_config = MCMCDriverConfig(
        k_alpha_proposal_var=0.0,
        k_sigma_proposal_var=0.0,
        obs_var_proposal_var=0.0,
    )

    lower = propose_params(
        MCMCParams(k_alpha=0.001, k_sigma=0.01, obs_var=1e-8),
        jax.random.PRNGKey(8),
        driver_config=driver_config,
        model_config=model_config,
    )
    upper = propose_params(
        MCMCParams(k_alpha=1.0, k_sigma=10.0, obs_var=1.0),
        jax.random.PRNGKey(9),
        driver_config=driver_config,
        model_config=model_config,
    )

    np.testing.assert_allclose(float(lower.k_alpha), 0.01)
    np.testing.assert_allclose(float(lower.k_sigma), 0.2)
    np.testing.assert_allclose(float(lower.obs_var), 1e-4)
    np.testing.assert_allclose(float(upper.k_alpha), 0.1)
    np.testing.assert_allclose(float(upper.k_sigma), 0.5)
    np.testing.assert_allclose(float(upper.obs_var), 1e-2)


def test_multi_chain_initial_states_are_distinct_and_seed_reproducible():
    dataset = _tiny_augmented_dataset()
    context = build_bffg_context(dataset, MCMCModelConfig(num_edge_steps=1))
    driver_config = MCMCDriverConfig(
        num_samples=1,
        num_chains=2,
        random_seed=123,
        progress_bar=False,
    )

    first = run_mcmc_chains(context, driver_config=driver_config)
    repeat = run_mcmc_chains(context, driver_config=driver_config)

    first_initials = np.array(
        [
            [float(result.initial_params.k_alpha), float(result.initial_params.k_sigma), float(result.initial_params.obs_var)]
            for result in first
        ]
    )
    repeat_initials = np.array(
        [
            [float(result.initial_params.k_alpha), float(result.initial_params.k_sigma), float(result.initial_params.obs_var)]
            for result in repeat
        ]
    )

    np.testing.assert_allclose(first_initials, repeat_initials)
    assert not np.allclose(first_initials[0], first_initials[1])


def test_multi_chain_initial_params_support_scalar_list_and_prior_defaults():
    dataset = _tiny_augmented_dataset()
    context = build_bffg_context(
        dataset,
        MCMCModelConfig(
            num_edge_steps=1,
            k_alpha_init=0.12,
            k_sigma_init=[0.21, 0.22, 0.23],
            obs_var_init=None,
        ),
    )
    driver_config = MCMCDriverConfig(
        num_samples=1,
        num_chains=3,
        random_seed=321,
        progress_bar=False,
    )

    results = run_mcmc_chains(context, driver_config=driver_config)

    keys = jax.random.split(jax.random.PRNGKey(driver_config.random_seed), driver_config.num_chains)
    expected_prior_obs_vars = []
    for chain_index in range(driver_config.num_chains):
        _, init_key = jax.random.split(keys[chain_index])
        expected_prior_obs_vars.append(
            float(
                sample_initial_params(
                    init_key,
                    driver_config=driver_config,
                    model_config=context.config,
                ).obs_var
            )
        )
    np.testing.assert_allclose([float(result.initial_params.k_alpha) for result in results], [0.12, 0.12, 0.12])
    np.testing.assert_allclose([float(result.initial_params.k_sigma) for result in results], [0.21, 0.22, 0.23])
    np.testing.assert_allclose([float(result.initial_params.obs_var) for result in results], expected_prior_obs_vars)


def test_multi_chain_initial_param_lists_must_match_num_chains():
    dataset = _tiny_augmented_dataset()
    context = build_bffg_context(
        dataset,
        MCMCModelConfig(
            num_edge_steps=1,
            k_alpha_init=[0.12],
        ),
    )
    driver_config = MCMCDriverConfig(
        num_samples=1,
        num_chains=2,
        progress_bar=False,
    )

    with pytest.raises(ValueError, match="k_alpha_init.*num_chains"):
        run_mcmc_chains(context, driver_config=driver_config)


def test_initial_param_values_must_be_finite_positive_scalars():
    dataset = _tiny_augmented_dataset()

    with pytest.raises(ValueError, match="obs_var_init.*finite positive"):
        build_bffg_context(
            dataset,
            MCMCModelConfig(
                num_edge_steps=1,
                obs_var_init=0.0,
            ),
        )


def test_process_backend_matches_sequential_chain_results():
    dataset = load_augmented_butterfly_tree(BUTTERFLIES_H5)
    model_config = MCMCModelConfig(
        num_edge_steps=1,
        k_sigma_max=0.2,
        covar_jitter=1e-6,
        dist_jitter=1e-7,
        enable_x64=True,
    )
    context = build_bffg_context(dataset, model_config)
    sequential_config = MCMCDriverConfig(
        num_samples=2,
        num_chains=2,
        pcn_eta=0.85,
        k_alpha_proposal_var=0.001,
        k_sigma_proposal_var=0.001,
        obs_var_proposal_var=0.001,
        random_seed=2025,
        progress_bar=False,
    )
    process_config = MCMCDriverConfig(
        num_samples=2,
        num_chains=2,
        pcn_eta=0.85,
        k_alpha_proposal_var=0.001,
        k_sigma_proposal_var=0.001,
        obs_var_proposal_var=0.001,
        random_seed=2025,
        progress_bar=False,
        chain_backend="process",
        num_processes=2,
    )

    sequential = run_mcmc_chains(context, driver_config=sequential_config, dataset_h5_path=BUTTERFLIES_H5)
    parallel = run_mcmc_chains(context, driver_config=process_config, dataset_h5_path=BUTTERFLIES_H5)

    assert len(parallel) == len(sequential) == 2
    for sequential_result, parallel_result in zip(sequential, parallel, strict=True):
        np.testing.assert_allclose(parallel_result.log_posteriors, sequential_result.log_posteriors)
        np.testing.assert_array_equal(parallel_result.accepted, sequential_result.accepted)
        for parameter_name in ("k_alpha", "k_sigma", "obs_var"):
            np.testing.assert_allclose(
                parallel_result.samples[parameter_name],
                sequential_result.samples[parameter_name],
            )
            np.testing.assert_allclose(
                float(getattr(parallel_result.initial_params, parameter_name)),
                float(getattr(sequential_result.initial_params, parameter_name)),
            )
            np.testing.assert_allclose(
                float(getattr(parallel_result.final_params, parameter_name)),
                float(getattr(sequential_result.final_params, parameter_name)),
            )
        np.testing.assert_allclose(parallel_result.final_noise, sequential_result.final_noise)


def test_process_backend_reports_per_chain_progress_from_main_process(monkeypatch):
    class FakeTqdm:
        instances = []

        def __init__(self, iterable=None, **kwargs):
            self.iterable = iterable
            self.kwargs = kwargs
            self.n = 0
            self.closed = False
            self.postfixes = []
            FakeTqdm.instances.append(self)

        def __iter__(self):
            for item in self.iterable:
                yield item

        def update(self, amount):
            self.n += amount

        def set_postfix(self, ordered_dict=None, **kwargs):
            values = dict(ordered_dict or {})
            values.update(kwargs)
            self.postfixes.append(values)

        def close(self):
            self.closed = True

    import src.driver as driver

    monkeypatch.setattr(driver, "tqdm", FakeTqdm)
    dataset = load_augmented_butterfly_tree(BUTTERFLIES_H5)
    context = build_bffg_context(
        dataset,
        MCMCModelConfig(
            num_edge_steps=1,
            k_sigma_max=0.2,
            covar_jitter=1e-6,
            dist_jitter=1e-7,
            enable_x64=True,
        ),
    )
    driver_config = MCMCDriverConfig(
        num_samples=1,
        num_chains=2,
        pcn_eta=0.85,
        k_alpha_proposal_var=0.001,
        k_sigma_proposal_var=0.001,
        obs_var_proposal_var=0.001,
        random_seed=2026,
        progress_bar=True,
        chain_backend="process",
        num_processes=2,
    )

    results = run_mcmc_chains(context, driver_config=driver_config, dataset_h5_path=BUTTERFLIES_H5)

    assert len(results) == 2
    assert [(bar.kwargs.get("desc"), bar.kwargs.get("position")) for bar in FakeTqdm.instances] == [
        ("chains", 0),
        ("chain 1/2", 1),
        ("chain 2/2", 2),
    ]
    assert [bar.kwargs.get("total") for bar in FakeTqdm.instances] == [2, 1, 1]
    assert [bar.n for bar in FakeTqdm.instances] == [2, 1, 1]
    assert FakeTqdm.instances[0].postfixes == []
    for chain_bar, result in zip(FakeTqdm.instances[1:], results, strict=True):
        assert chain_bar.postfixes[-1] == {
            "log_posterior": f"{result.log_posteriors[-1]:.3f}",
            "accept_rate": f"{result.acceptance_rate:.3f}",
        }
    assert all(bar.closed for bar in FakeTqdm.instances)


def test_process_backend_requires_dataset_h5_path():
    dataset = _tiny_augmented_dataset()
    context = build_bffg_context(dataset, MCMCModelConfig(num_edge_steps=1))
    driver_config = MCMCDriverConfig(
        num_samples=1,
        num_chains=1,
        chain_backend="process",
        progress_bar=False,
    )

    with pytest.raises(ValueError, match="dataset_h5_path"):
        run_mcmc_chains(context, driver_config=driver_config)


def test_driver_config_rejects_unknown_chain_backend():
    dataset = _tiny_augmented_dataset()
    context = build_bffg_context(dataset, MCMCModelConfig(num_edge_steps=1))
    driver_config = MCMCDriverConfig(
        num_samples=1,
        num_chains=1,
        chain_backend="thread",
        progress_bar=False,
    )

    with pytest.raises(ValueError, match="chain_backend"):
        run_mcmc_chains(context, driver_config=driver_config)


def test_driver_config_rejects_negative_num_processes():
    dataset = _tiny_augmented_dataset()
    context = build_bffg_context(dataset, MCMCModelConfig(num_edge_steps=1))
    driver_config = MCMCDriverConfig(
        num_samples=1,
        num_chains=1,
        num_processes=-1,
        progress_bar=False,
    )

    with pytest.raises(ValueError, match="num_processes"):
        run_mcmc_chains(context, driver_config=driver_config)


def test_run_mcmc_writes_full_chain_log_hdf5(tmp_path):
    dataset = _tiny_augmented_dataset_with_phylo_root()
    run_dir = tmp_path / "tiny-run"

    model_config = MCMCModelConfig(num_edge_steps=1)
    driver_config = MCMCDriverConfig(
        num_samples=3,
        num_chains=2,
        random_seed=123,
        progress_bar=False,
    )
    result = run_mcmc(
        dataset,
        run_dir=run_dir,
        model_config=model_config,
        driver_config=driver_config,
    )

    assert result.run_dir == run_dir
    for filename in ["config.json", "summary.json", "artifacts.h5"]:
        path = run_dir / filename
        assert path.exists()
        assert path.stat().st_size > 0

    with h5py.File(run_dir / "artifacts.h5", "r") as h5:
        assert set(h5["samples"]) == {"k_alpha", "k_sigma", "obs_var", "phylo_root"}
        assert set(h5["trace"]) == {"accepted", "log_posteriors"}
        for parameter_name in ("k_alpha", "k_sigma", "obs_var"):
            assert set(h5[f"samples/{parameter_name}"]) == {"chain_000", "chain_001"}
            assert h5[f"samples/{parameter_name}/chain_000"].shape == (3,)
            assert h5[f"samples/{parameter_name}/chain_001"].shape == (3,)
        assert set(h5["samples/phylo_root"]) == {"chain_000", "chain_001"}
        assert h5["samples/phylo_root/chain_000"].shape == (3, 2, 2)
        assert h5["samples/phylo_root/chain_001"].shape == (3, 2, 2)
        assert h5["trace/log_posteriors/chain_000"].shape == (3,)
        assert h5["trace/log_posteriors/chain_001"].shape == (3,)
        assert h5["trace/accepted/chain_000"].shape == (3,)
        assert h5["trace/accepted/chain_001"].shape == (3,)
        np.testing.assert_allclose(h5["samples/k_alpha/chain_000"][:], result.samples["k_alpha"][0])
        np.testing.assert_allclose(h5["samples/phylo_root/chain_000"][:], result.samples["phylo_root"][0])
        np.testing.assert_allclose(h5["trace/log_posteriors/chain_000"][:], result.log_posteriors[0])
        np.testing.assert_array_equal(h5["trace/accepted/chain_000"][:], result.accepted[0])

    assert result.summary["num_chains"] == 2
    assert result.summary["num_samples"] == 3
    assert result.summary["total_iterations"] == 3
    assert result.summary["artifact_file"] == "artifacts.h5"
    assert len(result.summary["initial_params"]) == 2
    keys = jax.random.split(jax.random.PRNGKey(driver_config.random_seed), driver_config.num_chains)
    expected_initial_params = []
    for chain_index in range(driver_config.num_chains):
        _, init_key = jax.random.split(keys[chain_index])
        expected_initial_params.append(
            sample_initial_params(
                init_key,
                driver_config=driver_config,
                model_config=model_config,
            )
        )
    for actual, expected in zip(result.summary["initial_params"], expected_initial_params, strict=True):
        np.testing.assert_allclose(actual["k_alpha"], float(expected.k_alpha))
        np.testing.assert_allclose(actual["k_sigma"], float(expected.k_sigma))
        np.testing.assert_allclose(actual["obs_var"], float(expected.obs_var))
    assert not np.allclose(
        list(result.summary["initial_params"][0].values()),
        list(result.summary["initial_params"][1].values()),
    )


def test_run_mcmc_overwrites_existing_run_directory(tmp_path):
    dataset = _tiny_augmented_dataset_with_phylo_root()
    run_dir = tmp_path / "same-name-run"
    run_dir.mkdir()
    stale_file = run_dir / "stale.txt"
    stale_file.write_text("old output\n")

    run_mcmc(
        dataset,
        run_dir=run_dir,
        model_config=MCMCModelConfig(num_edge_steps=1),
        driver_config=MCMCDriverConfig(
            num_samples=1,
            num_chains=1,
            random_seed=123,
            progress_bar=False,
        ),
    )

    assert not stale_file.exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "artifacts.h5").exists()


def test_run_artifacts_prepare_run_dir_replaces_stale_directory(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    stale_file = run_dir / "stale.txt"
    stale_file.write_text("old output\n")

    prepared = prepare_run_dir(run_dir)

    assert prepared == run_dir
    assert prepared.exists()
    assert not stale_file.exists()


def _tiny_augmented_dataset():
    from hyperiax import Topology, Tree

    from src.loader import AugmentedButterflyTree

    topology = Topology.from_parents(
        np.array([0, 0, 0], dtype=np.int32),
        names=("super_root", "leaf_a", "leaf_b"),
    )
    coords = jnp.array(
        [
            [[0.0, 0.0], [0.4, 0.0]],
            [[-0.1, 0.0], [0.35, 0.05]],
            [[0.1, 0.0], [0.45, -0.05]],
        ],
        dtype=jnp.float32,
    )
    tree = Tree.from_data(
        topology,
        {
            "coords": coords,
            BFFG_FIELDS.edge_len: jnp.array([0.0, 1.0, 1.0], dtype=jnp.float32),
            "is_hidden": jnp.array([False, False, False]),
            "is_observed": jnp.array([True, True, True]),
        },
    )
    return AugmentedButterflyTree(
        tree=tree,
        node_names=("super_root", "leaf_a", "leaf_b"),
        node_types=("super_root", "leaf", "leaf"),
        leaf_names=("leaf_a", "leaf_b"),
        newick_aug="(leaf_a:1,leaf_b:1)super_root;",
        h5_path=Path("tiny.h5"),
        normalization_center=np.zeros(2),
        normalization_scale=1.0,
        edge_len_normalizer=1.0,
    )


def _tiny_augmented_dataset_with_phylo_root():
    from hyperiax import Topology, Tree

    from src.loader import AugmentedButterflyTree

    topology = Topology.from_parents(
        np.array([0, 0, 1, 1], dtype=np.int32),
        names=("super_root", "phylo_root", "leaf_a", "leaf_b"),
    )
    coords = jnp.array(
        [
            [[0.0, 0.0], [0.4, 0.0]],
            [[jnp.nan, jnp.nan], [jnp.nan, jnp.nan]],
            [[-0.1, 0.0], [0.35, 0.05]],
            [[0.1, 0.0], [0.45, -0.05]],
        ],
        dtype=jnp.float32,
    )
    tree = Tree.from_data(
        topology,
        {
            "coords": coords,
            BFFG_FIELDS.edge_len: jnp.array([0.0, 0.5, 1.0, 1.0], dtype=jnp.float32),
            "is_hidden": jnp.array([False, True, False, False]),
            "is_observed": jnp.array([True, False, True, True]),
        },
    )
    return AugmentedButterflyTree(
        tree=tree,
        node_names=("super_root", "phylo_root", "leaf_a", "leaf_b"),
        node_types=("super_root", "phylo_root", "leaf", "leaf"),
        leaf_names=("leaf_a", "leaf_b"),
        newick_aug="((leaf_a:1,leaf_b:1)phylo_root:0.5)super_root;",
        h5_path=Path("tiny.h5"),
        normalization_center=np.zeros(2),
        normalization_scale=1.0,
        edge_len_normalizer=1.0,
    )


def _tiny_augmented_dataset_3d_with_phylo_root():
    from hyperiax import Topology, Tree

    from src.loader import AugmentedButterflyTree

    topology = Topology.from_parents(
        np.array([0, 0, 1, 1], dtype=np.int32),
        names=("super_root", "phylo_root", "leaf_a", "leaf_b"),
    )
    coords = jnp.array(
        [
            [[0.0, 0.0, 0.0], [0.4, 0.0, 0.2]],
            [[jnp.nan, jnp.nan, jnp.nan], [jnp.nan, jnp.nan, jnp.nan]],
            [[-0.1, 0.0, 0.1], [0.35, 0.05, 0.2]],
            [[0.1, 0.0, -0.1], [0.45, -0.05, 0.4]],
        ],
        dtype=jnp.float32,
    )
    tree = Tree.from_data(
        topology,
        {
            "coords": coords,
            BFFG_FIELDS.edge_len: jnp.array([0.0, 0.5, 1.0, 1.0], dtype=jnp.float32),
            "is_hidden": jnp.array([False, True, False, False]),
            "is_observed": jnp.array([True, False, True, True]),
        },
    )
    return AugmentedButterflyTree(
        tree=tree,
        node_names=("super_root", "phylo_root", "leaf_a", "leaf_b"),
        node_types=("super_root", "phylo_root", "leaf", "leaf"),
        leaf_names=("leaf_a", "leaf_b"),
        newick_aug="((leaf_a:1,leaf_b:1)phylo_root:0.5)super_root;",
        h5_path=Path("tiny-3d.h5"),
        normalization_center=np.zeros(3),
        normalization_scale=1.0,
        edge_len_normalizer=1.0,
    )


def _direct_tiny_log_posterior(root_value, leaf_values, params, noise, config):
    n_landmarks = 2
    d_landmarks = 2
    eye = jnp.eye(n_landmarks, dtype=leaf_values.dtype)
    h_leaf = eye / params.obs_var
    sigma_obs = params.obs_var * eye

    leaf_ptnl = jax.vmap(lambda v: dot_factorized(h_leaf, v))(leaf_values)
    leaf_prec = jnp.repeat(h_leaf[None, :, :], leaf_values.shape[0], axis=0)
    leaf_tildea = jax.vmap(
        lambda v: covariance_matrix(
            v,
            params,
            covar_jitter=config.covar_jitter,
            dist_jitter=config.dist_jitter,
        )
    )(leaf_values)

    def child_pullback(edge_len, ptnl_v, prec_v, tildea_v):
        ts = jnp.linspace(0.0, edge_len, config.num_edge_steps + 1, dtype=leaf_values.dtype)

        def per_t(t):
            phi_inv = eye + prec_v @ tildea_v * (edge_len - t)
            prec_t = solve_factorized(phi_inv, prec_v).reshape(prec_v.shape)
            ptnl_t = solve_factorized(phi_inv, ptnl_v).reshape(ptnl_v.shape)
            return prec_t, ptnl_t

        precs, ptnls = jax.vmap(per_t)(ts)
        v_T = solve_factorized(prec_v, ptnl_v)
        log_norm_0 = jax.vmap(
            lambda coordinate_values: log_gaussian_prec(
                jnp.zeros(n_landmarks, dtype=leaf_values.dtype),
                coordinate_values,
                precs[0],
            ),
            in_axes=1,
        )(v_T.reshape((n_landmarks, d_landmarks))).sum()
        return log_norm_0, ptnls, precs

    child_log_norm, child_ptnls, child_precs = jax.vmap(child_pullback)(
        jnp.ones((2,), dtype=leaf_values.dtype),
        leaf_ptnl,
        leaf_prec,
        leaf_tildea,
    )

    root_log_norm = child_log_norm.sum()
    root_ptnl = child_ptnls[:, 0].sum(axis=0)
    root_prec = child_precs[:, 0].sum(axis=0)
    root_log_likelihood = root_log_norm + root_ptnl @ root_value - 0.5 * root_value @ dot_factorized(
        root_prec,
        root_value,
    )

    log_corrs = []
    endpoints = []
    for index in range(2):
        values, log_corr = forward_guided(
            root_value,
            child_precs[index],
            child_ptnls[index],
            leaf_tildea[index],
            jnp.ones((1,), dtype=leaf_values.dtype),
            noise[index + 1],
            params,
            covar_jitter=config.covar_jitter,
            dist_jitter=config.dist_jitter,
        )
        endpoints.append(values[-1])
        log_corrs.append(log_corr)

    residuals = jnp.stack(endpoints) - leaf_values
    leaf_log_likelihood = normal_logpdf(residuals, 0.0, jnp.sqrt(params.obs_var)).mean()
    log_prior = (
        inverse_gamma_logpdf(params.k_alpha, config.k_alpha_prior_alpha, config.k_alpha_prior_beta)
        + inverse_gamma_logpdf(params.k_sigma, config.k_sigma_prior_alpha, config.k_sigma_prior_beta)
        + inverse_gamma_logpdf(params.obs_var, config.obs_var_prior_alpha, config.obs_var_prior_beta)
    )

    return log_prior + root_log_likelihood + jnp.stack(log_corrs).mean() + leaf_log_likelihood
