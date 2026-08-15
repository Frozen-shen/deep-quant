import { Component, type ReactNode } from 'react'
import { Layout, Menu, Alert, Button, Select, Typography } from 'antd'
import { Routes, Route, Link, useLocation } from 'react-router-dom'
import Overview from './pages/Overview'
import Portfolio from './pages/Portfolio'
import Signals from './pages/Signals'
import Factors from './pages/Factors'
import Experiments from './pages/Experiments'
import Stocks from './pages/Stocks'
import DataStatus from './pages/DataStatus'
import Trading from './pages/Trading'
import { ExperimentProvider, useExperiment } from './experiment-context'

// 四区导航: 仪表盘 / 研究(因子+实验) / 交易(组合+信号+成交) / 数据(个股+数据状态)
const items = [
  { key: '/', label: <Link to="/">仪表盘</Link> },
  { key: '/research', label: '研究', children: [
    { key: '/factors', label: <Link to="/factors">因子</Link> },
    { key: '/experiments', label: <Link to="/experiments">实验</Link> },
  ]},
  { key: '/trading', label: '交易', children: [
    { key: '/portfolio', label: <Link to="/portfolio">组合</Link> },
    { key: '/signals', label: <Link to="/signals">信号</Link> },
    { key: '/trades', label: <Link to="/trades">成交明细</Link> },
  ]},
  { key: '/data', label: '数据', children: [
    { key: '/stocks', label: <Link to="/stocks">个股</Link> },
    { key: '/datastatus', label: <Link to="/datastatus">数据状态</Link> },
  ]},
]

/** 全局错误边界：页面渲染异常时兜底提示（数据请求失败已由各页 isError 分支处理）。 */
class ErrorBoundary extends Component<{ children: ReactNode }, { hasError: boolean; message: string }> {
  constructor(props: { children: ReactNode }) {
    super(props)
    this.state = { hasError: false, message: '' }
  }
  static getDerivedStateFromError(error: unknown) {
    return { hasError: true, message: error instanceof Error ? error.message : String(error) }
  }
  render() {
    if (this.state.hasError) {
      return (
        <Alert type="error" showIcon message="页面渲染出错"
          description={`${this.state.message}。请检查后端是否已启动 (uvicorn :8000)，或刷新重试`}
          action={<Button size="small" onClick={() => { this.setState({ hasError: false }); window.location.reload() }}>刷新</Button>} />
      )
    }
    return this.props.children
  }
}

/** Header 实验选择器（全局上下文，URL ?exp= 同步）。 */
function ExperimentPicker() {
  const { expId, setExpId, registry } = useExperiment()
  const exps = registry?.experiments ?? []
  return (
    <Select
      style={{ width: 340 }}
      placeholder="选择实验"
      value={expId ?? undefined}
      onChange={(v) => setExpId(v)}
      options={exps.map(e => ({ value: e.id, label: `${e.name}${e.kind === 'walkforward' ? ' (回测)' : ' (实验)'}` }))}
    />
  )
}

function Shell() {
  const location = useLocation()
  const selectedKey = location.pathname.startsWith('/trades') ? '/trades' : location.pathname
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Layout.Header style={{ display: 'flex', alignItems: 'center', gap: 16, padding: '0 24px' }}>
        <Typography.Text strong style={{ color: '#fff', fontSize: 16 }}>quant-starter</Typography.Text>
        <ExperimentPicker />
      </Layout.Header>
      <Layout>
        <Layout.Sider theme="dark" width={180}>
          <Menu theme="dark" mode="inline" items={items} selectedKeys={[selectedKey]} defaultOpenKeys={['/research', '/trading', '/data']} />
        </Layout.Sider>
        <Layout.Content style={{ padding: 24 }}>
          <ErrorBoundary>
            <Routes>
              <Route path="/" element={<Overview />} />
              <Route path="/factors" element={<Factors />} />
              <Route path="/experiments" element={<Experiments />} />
              <Route path="/portfolio" element={<Portfolio />} />
              <Route path="/signals" element={<Signals />} />
              <Route path="/trades" element={<Trading />} />
              <Route path="/stocks" element={<Stocks />} />
              <Route path="/datastatus" element={<DataStatus />} />
            </Routes>
          </ErrorBoundary>
        </Layout.Content>
      </Layout>
    </Layout>
  )
}

export default function App() {
  return (
    <ExperimentProvider>
      <Shell />
    </ExperimentProvider>
  )
}
