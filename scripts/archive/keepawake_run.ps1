# 防休眠: 全量回测期间保持系统唤醒 (ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED | ES_AWAYMODE_REQUIRED)
# 注: 本脚本是被 active 回测流程的运维伴生, 跑完即停; 不参与任何计算
Add-Type -Namespace W -Name KA -MemberDefinition '[DllImport("kernel32.dll")] public static extern uint SetThreadExecutionState(uint es);'
while ($true) {
    [W.KA]::SetThreadExecutionState(0xC0000001) | Out-Null
    Start-Sleep -Seconds 30
}
