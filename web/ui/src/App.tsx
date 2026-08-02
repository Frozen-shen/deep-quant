import { Layout, Menu } from 'antd'
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

export default function App() {
  return (
    <Layout style={{ minHeight: '100vh' }}>
      <Layout.Sider theme="dark">
        <div style={{ color: '#fff', padding: 16, fontWeight: 600 }}>quant-starter</div>
        <Menu theme="dark" mode="inline" items={items} defaultSelectedKeys={['/']} />
      </Layout.Sider>
      <Layout.Content style={{ padding: 24 }}>
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
      </Layout.Content>
    </Layout>
  )
}
