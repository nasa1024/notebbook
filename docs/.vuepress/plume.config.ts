import { defineThemeConfig } from 'vuepress-theme-plume'
import { enNavbar, zhNavbar } from './navbar'
import { enNotes, zhNotes } from './notes'

/**
 * @see https://theme-plume.vuejs.press/config/basic/
 */
export default defineThemeConfig({
  logo: 'https://theme-plume.vuejs.press/plume.png',
  // your git repo url
  docsRepo: '',
  docsDir: 'docs',

  appearance: true,

  social: [
    { icon: 'github', link: 'https://github.com/nasa1024' },
    { icon: 'twitter', link: 'https://x.com/CindyCo98345620' },
  ],

  locales: {
    '/': {
      profile: {
        avatar: 'https://avatars.githubusercontent.com/u/28981872?s=400&u=62aa62940002d7ce7a149dd856809208e8c34116&v=4',
        name: 'nasa1024',
        description: '全栈开发工程师',
        circle: true,
        // location: '',
        // organization: '',
      },

      navbar: zhNavbar,
      notes: zhNotes,
    },
    '/en/': {
      profile: {
        avatar: 'https://theme-plume.vuejs.press/plume.png',
        name: 'nasa1024',
        description: 'Full Stack Developer',
        // circle: true,
        // location: '',
        // organization: '',
      },

      navbar: enNavbar,
      notes: enNotes,
    },
  },
})
