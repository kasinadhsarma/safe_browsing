import { Providers } from "@/app/providers"

export default function TechnologyLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <Providers>
      {children}
    </Providers>
  )
}