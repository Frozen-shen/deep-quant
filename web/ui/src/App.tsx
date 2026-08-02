import { Component, type ReactNode } from 'react'
import { Layout, Menu, Alert, Button } from 'antd'
import { Routes, Route, Link } from 'react-router-dom'
import Overview from './pages/Overview'
import Portfolio from './pages/Portfolio'
import Signals from './pages/Signals'
import Factors from './pages/Factors'
import Experiments from './pages/Experiments'
import Stocks from './pages/Stocks'
import DataStatus from './pages/DataStatus'
import Trading from './pages/Trading'

const items = [
  { key: '/', label: <Link to="/">总览</Link> },
  { key: '/portfolio', label: <Link to="/portfolio">组合</Link> },
  { key: '/signals', label: <Link to="/signals">信号</Link> },
  { key: '/factors', label: <Link to="/factors">因子</Link> },
  { key: '/experiments', label: <Link to="/experiments">实验</Link> },
  { key: '/stocks', label: <Link to="/stocks">个股</Link> },
  { key: '/trading', label: <Link to="/trading">交易监控</Link> },
  { key: '/data', label: <Link to="/data">数据状态</Link> },
]

/** 全局错误边界：页面渲染异常时兜底提示（数据请求失败已由各页 isError 分支处理）。 */
class ErrorBoundary extends Component<{ children: ReactNode }, { hasError: boolean }> {
  constructor(props: { children: ReactNode }) {
    super(props)
    this.state = { hasError: false }
  }
  static getDerivedStateFromError() {
    return { hasError: true }
  }
  render() {
    if (this.state.hasError) {
      return (
        <Alert type="error" showIcon message="页面渲染出错"
          description="请检查后端是否已启动 (uvicorn :8000)，或刷新重试"
          action={<Button size="small" onClick={() => { this.setState({ hasError: false }); window.location.reload() }}>刷新</Button>} />
      )
    }
    return this.props.children
  }
}

export default function App() {
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Layout.Sider theme="dark">
        <div style={{ color: '#fff', padding: 16, fontWeight: 600 }}>quant-starter</div>
        <Menu theme="dark" mode="inline" items={items} defaultSelectedKeys={['/']} />
      </Layout.Sider>
      <Layout.Content style={{ padding: 24 }}>
        <ErrorBoundary>
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/portfolio" element={<Portfolio />} />
            <Route path="/signals" element={<Signals />} />
            <Route path="/factors" element={<Factors />} />
            <Route path="/experiments" element={<Experiments />} />
            <Route path="/stocks" element={<Stocks />} />
            <Route path="/trading" element={<Trading />} />
            <Route path="/data" element={<DataStatus />} />
          </Routes>
        </ErrorBoundary>
      </Layout.Content>
    </Layout>
  )
}
