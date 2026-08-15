import { createContext, useContext, useEffect, useMemo, type ReactNode } from 'react'
import { useSearchParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import { fetchExperimentRegistry, fetchExperimentDetail, type ExperimentDetail, type ExperimentRegistryItem } from './api'

interface ExperimentCtx {
  expId: string | null
  setExpId: (id: string | null) => void
  registry: { count: number; experiments: ExperimentRegistryItem[] } | undefined
  detail: ExperimentDetail | undefined
  detailLoading: boolean
}

const Ctx = createContext<ExperimentCtx | null>(null)

export function ExperimentProvider({ children }: { children: ReactNode }) {
  const [params, setParams] = useSearchParams()
  const expId = params.get('exp')

  const registry = useQuery({ queryKey: ['exp-registry'], queryFn: fetchExperimentRegistry, staleTime: 60_000 })
  const detail = useQuery({
    queryKey: ['exp-detail', expId],
    queryFn: () => fetchExperimentDetail(expId as string),
    enabled: !!expId,
    staleTime: 60_000,
  })

  // 默认选中最新实验
  useEffect(() => {
    if (!expId && registry.data?.experiments?.length) {
      const latest = registry.data.experiments[0]
      setParams(prev => { const p = new URLSearchParams(prev); p.set('exp', latest.id); return p })
    }
  }, [expId, registry.data, setParams])

  const value = useMemo(() => ({
    expId,
    setExpId: (id: string | null) => {
      setParams(prev => {
        const p = new URLSearchParams(prev)
        if (id) p.set('exp', id); else p.delete('exp')
        return p
      })
    },
    registry: registry.data,
    detail: detail.data,
    detailLoading: detail.isLoading,
  }), [expId, registry.data, detail.data, detail.isLoading, setParams])

  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}

export function useExperiment(): ExperimentCtx {
  const v = useContext(Ctx)
  if (!v) throw new Error('useExperiment must be used within ExperimentProvider')
  return v
}
