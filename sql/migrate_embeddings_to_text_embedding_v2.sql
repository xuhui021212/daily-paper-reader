-- Migrate paper embedding columns from BAAI/bge-small-en-v1.5 (384d)
-- to AGICTO/Alibaba text-embedding-v2 (1536d).
--
-- Run this once in the Supabase SQL Editor before running the
-- "migrate-embeddings" GitHub Actions workflow.
--
-- This intentionally clears old embeddings because 384-dimensional vectors
-- cannot be compared with 1536-dimensional vectors.

create extension if not exists vector;

do $$
declare
  table_name text;
  tables text[] := array[
    'arxiv_papers',
    'biorxiv_papers',
    'medrxiv_papers',
    'chemrxiv_papers',
    'neurips_openreview_papers',
    'iclr_openreview_papers',
    'icml_openreview_papers',
    'acl_papers',
    'emnlp_papers',
    'aaai_papers',
    'papers'
  ];
begin
  foreach table_name in array tables loop
    if to_regclass(format('public.%I', table_name)) is null then
      raise notice 'skip missing table public.%', table_name;
      continue;
    end if;

    execute format('drop index if exists public.%I', table_name || '_embedding_hnsw_idx');
    execute format(
      'update public.%I
       set embedding = null,
           embedding_model = null,
           embedding_dim = null,
           embedding_updated_at = null
       where embedding is not null',
      table_name
    );
    execute format('alter table public.%I alter column embedding type vector(1536)', table_name);
    execute format(
      'create index if not exists %I on public.%I using hnsw (embedding vector_cosine_ops)',
      table_name || '_embedding_hnsw_idx',
      table_name
    );

    raise notice 'migrated public.% embedding column to vector(1536)', table_name;
  end loop;
end $$;
